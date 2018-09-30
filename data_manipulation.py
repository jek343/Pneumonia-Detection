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

ORIG_SIZE = 1024
DATA_DIR = "data" #TODO fill in the data directory you're using
ROOT_DIR = "lung_opacity_detection" #TODO fill in the directory this file is in
PICKLED_TRAIN = "dataset_train.obj"
PICKLED_VALID = "dataset_val.obj"
TRAIN_DICOM_DIR = os.path.join(DATA_DIR, 'stage_1_train_images')

from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import utils
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log

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

def datset_split(exists=True):
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
