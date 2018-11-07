#!/usr/bin/env python
# coding: utf-8

#import data_manipulation
import pickle
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.exposure import adjust_sigmoid
from skimage import feature
from skimage.color import rgb2gray


with open("data/dataset_train.obj", "rb") as f:
    train = pickle.load(f)


def prior():
    ids = range(train.size())
    #sample = random.sample(ids, 20)
    cnt_box = np.zeros([1024,1024])

    p_imgs = [] #build up list of masks (numpy array) of images with pneumonia
    hasPneu = 0
    for id in ids[:]:
        mask, class_ids = train.load_mask(id) #mask = numpy array of boolean values. 1 if pneumonia in that pixel
                   #class_ids = 1 elt array. 1 if image has pneumonia. 0 otherwise
        if (class_ids.all()):
            hasPneu += 1
            cnt_box = cnt_box + np.sum(mask.astype(int), axis = 2)
    cnt_box = cnt_box / hasPneu
    #x = np.max(cnt_box)
    return cnt_box

if __name__ == "__main__":
    cnt_box = prior()
    np.save("Prior", cnt_box)

