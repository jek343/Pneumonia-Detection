import cv2
import glob
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import pandas as pd
import pickle
import random
import sys

from skimage import feature
from skimage.color import rgb2gray
from skimage.exposure import adjust_sigmoid
import skimage.transform

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log

# ================= DIRECTORIES =================

DATA_DIR = 'C:\\Users\\katie\\Documents\\DataScience\\PneumoniaDetection\\dataSet'
DATA_DIR_OBJS = "C:\\Users\\katie\\Documents\\DataScience\\PneumoniaDetection\\data"
#ROOT_DIR = 'C:\\Users\\katie\\Documents\\DataScience\\PneumoniaDetection'
PICKLED_TRAIN = "dataset_train.obj"
PICKLED_VALID = "dataset_val.obj"
TRAIN_DICOM_DIR = os.path.join(DATA_DIR, 'stage_1_train_images')

# ================= CONSTANTS =================
ORIG_SIZE = 1024
RESIZED_SIZE = 500
SMALL_TRAIN_SIZE = 1000
SMALL_VAL_SIZE = 800

class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # add images
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def size(self):
        return len(self.image_info)

    def labels_list(self):
        return [self.load_mask(i)[1][0] for i in range(self.size())]

def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        #existing errors in #images in folder vs. stage_1.csv
        try:
            fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
            image_annotations[fp].append(row)
        except KeyError:
            pass
    return image_fps, image_annotations

def dataset_split(exists=True):
    train_pickle = os.path.join(DATA_DIR, PICKLED_TRAIN)
    valid_pickle = os.path.join(DATA_DIR, PICKLED_VALID)

    if os.path.exists(train_pickle) and exists:
        raise ValueError("Pickles for the Trainig and Validation Set exist. Set exists to False,\
                to overwrite them.")

    anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
    image_fps, image_annotations = parse_dataset(TRAIN_DICOM_DIR, anns=anns)
    image_fps_list = list(image_fps)
    random.seed(42)
    random.shuffle(image_fps_list)
    val_size = 1500
    image_fps_val = image_fps_list[:val_size]
    image_fps_train = image_fps_list[val_size:]

    #training set
    dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_train.prepare()

    #validation set
    dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
    dataset_val.prepare()

    with open(train_pickle, "wb") as p_train:
        pickle.dump(dataset_train, p_train)

    with open(valid_pickle, "wb") as p_val:
        pickle.dump(dataset_val, p_val)

def load_dataset():
    train_pickle = os.path.join(DATA_DIR, PICKLED_TRAIN)
    valid_pickle = os.path.join(DATA_DIR, PICKLED_VALID)
    with open(train_pickle, "rb") as f:
        train_detector = pickle.load(f)

    with open(valid_pickle, "rb") as f:
        valid_detector = pickle.load(f)

    return train_detector, valid_detector

''' Generates a subset of the trainin dataset, held by detector.
    For effective use, SMALL_TRAIN_SIZE must not be greater than twice
    the number of positive examples in the dataset. '''
def small_train_matrix(detector, thresholded=False, edge_detected=False, e_sigma=0.5):
    pn_remaining = int((SMALL_TRAIN_SIZE)/2)
    not_pn_remaining = int((SMALL_TRAIN_SIZE)/2)
    img_id = 0

    train_x = np.zeros((SMALL_TRAIN_SIZE, RESIZED_SIZE, RESIZED_SIZE))
    train_y = np.zeros((SMALL_TRAIN_SIZE, ))
    train_id = 0

    while pn_remaining != 0 or not_pn_remaining != 0:
        image = detector.load_image(img_id)
        class_id = detector.load_mask(img_id)[1][0]

        if (class_id == 1 and pn_remaining > 0) or (class_id == 0 and not_pn_remaining > 0):
            if thresholded:
                image = adjust_sigmoid(image)

            image = rgb2gray(image)

            if edge_detected:
                image = feature.canny(image, sigma = e_sigma)

            train_x[train_id, :, :] = skimage.transform.resize(image, output_shape = (RESIZED_SIZE, RESIZED_SIZE))
            train_y[train_id] = class_id
            train_id += 1

            if class_id == 1:
                pn_remaining -= 1
            else:
                not_pn_remaining -= 1
        img_id += 1

    return train_x, train_y

def val_train_matrix(detector):
    val_x = np.zeros((SMALL_VAL_SIZE, RESIZED_SIZE, RESIZED_SIZE))
    val_y = np.zeros((SMALL_VAL_SIZE, ))
    val_id = 0

    for img_id in range(SMALL_VAL_SIZE):
        image = rgb2gray(detector.load_image(img_id))
        image = skimage.transform.resize(image, output_shape = (RESIZED_SIZE, RESIZED_SIZE))
        class_id = detector.load_mask(img_id)[1][0]

        val_x[val_id, :, :] = image
        val_y[val_id] = class_id

    return val_x, val_y
