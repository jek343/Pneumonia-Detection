import tensorflow as tf
import keras as k
from keras import initializers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten
from keras.models import Model
<<<<<<< HEAD
from keras import initializers
=======

>>>>>>> 2d860b3b1346ace672dcb1662c9e3909407d2c2e
import cv2
import glob
import json
import math
import matplotlib.pyplot as plt
import os
import pydicom
import pandas as pd
import random
import sys

import data_manipulation
import data_generator

# ========== CONSTANTS ========== START
IMAGE_DIM = 128
PARAMS = {'dim': (IMAGE_DIM,IMAGE_DIM),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# ========== CONSTANTS ========== END

def create_classifier():
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (IMAGE_DIM, IMAGE_DIM, 3))

    class_maxpool1 = MaxPool2D(pool_size=(2,2), strides = None, padding='same', name = 'classifier_maxpool1')(vgg16.layers[-1].output)
    class_conv2d_1 = Conv2D(filters = 256, kernel_size=(1,1), padding='same', name='classifier_conv2d_1',
                        activation='relu')(class_maxpool1)
    class_conv2d_2 = Conv2D(filters = 56, kernel_size=(1,1), padding='same', name='classifier_conv2d_2',
                        activation='relu')(class_conv2d_1)
    class_conv2d_3 = Conv2D(filters = 28, kernel_size=(1,1), padding='same', name='classifier_conv2d_3',
                        activation='relu')(class_conv2d_2)

    flatten_1 = Flatten()(class_conv2d_3)
    class_dense_1 = Dense(units = 2, activation ='softmax')(flatten_1)

    return Model(input=vgg16.input, output = class_dense_1)
''' train_set: DetectorDataset object for the training set.
    val_set  : DetectorDataset object for the validation set. '''
def train_classifier(model, train_set, val_set):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    train_labels = train.labels_list()
    val_labels = val.labels_list()

    training_generator = data_generator.DataGenerator(train, range(train.size()), train_labels, **PARAMS)
    validation_generator = data_generator.DataGenerator(val, range(val.size()), val_labels, **PARAMS)

    model.fit_generator(generator=training_generator,
                           validation_data=validation_generator,
                           use_multiprocessing=False)

def main():
    model = create_classifier()
    train, val = load_dataset()
    train_classifier(model, train, val)
