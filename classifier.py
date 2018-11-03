import tensorflow as tf
import keras as k
from keras import initializers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications import resnet50
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

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

import data_manipulation
import data_generator

# ========== CONSTANTS ========== START
IMAGE_DIM = 1024
PARAMS = {'dim': (IMAGE_DIM,IMAGE_DIM),
          'batch_size': 3,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

# ========== CONSTANTS ========== END

def create_classifier():
    vgg16 = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape = (IMAGE_DIM, IMAGE_DIM, 3))
    #for layer in vgg16.layers:
    #    layer.trainable = False

    class_maxpool1 = MaxPool2D(pool_size=(2,2), strides = None, padding='same', name = 'classifier_maxpool1')(vgg16.layers[-40].output)
    class_conv2d_1 = Conv2D(filters = 64, kernel_size=(1,1), padding='same', name='classifier_conv2d_1',
                        activation='relu')(class_maxpool1)
    class_conv2d_2 = Conv2D(filters = 32, kernel_size=(1,1), padding='same', name='classifier_conv2d_2',
                        activation='relu')(class_conv2d_1)
    class_conv2d_3 = Conv2D(filters = 16, kernel_size=(1,1), padding='same', name='classifier_conv2d_3',
                        activation='relu')(class_conv2d_2)

    flatten_1 = Flatten()(class_conv2d_3)
    class_dense_1 = Dense(units = 2, activation ='softmax')(flatten_1)

    return Model(input=vgg16.input, output = class_dense_1)
''' train_set: DetectorDataset object for the training set.
    val_set  : DetectorDataset object for the validation set. '''
def train_classifier(model, train_set, val_set):
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    train_labels = train_set.labels_list()
    val_labels = val_set.labels_list()

    training_generator = data_generator.DataGenerator(train_set, range(train_set.size()), train_labels, **PARAMS)
    validation_generator = data_generator.DataGenerator(val_set, range(val_set.size()), val_labels, **PARAMS)
    filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit_generator(generator=training_generator,
                           validation_data=validation_generator,
                           use_multiprocessing=True, epochs = 20, callbacks = [checkpoint])
    model.save("resnetsaveus.h5")

def main():
    model = create_classifier()
    train, val = data_manipulation.load_dataset()
    train_classifier(model, train, val)

if __name__ == '__main__':
    main()
