import numpy as np
import tensorflow as tf
import keras as k
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten
from keras.models import Model
from keras import initializers

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
import pickle

# ========== CONSTANTS ========== #
IMAGE_DIM = 128

with open("data/dataset_train.obj", "rb") as f:
    train = pickle.load(f)
with open("data/dataset_val.obj", "rb") as f:
    val = pickle.load(f)

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

my_model = Model(input=vgg16.input, output = class_dense_1)

print(my_model.summary())

my_model.compile(loss='categorical_crossentropy', optimizer='adam')
train_labels = train.labels_list()
val_labels = val.labels_list()

params = {'dim': (IMAGE_DIM,IMAGE_DIM),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}
training_generator = data_generator.DataGenerator(train, range(train.size()), train_labels, **params)
validation_generator = data_generator.DataGenerator(val, range(val.size()), val_labels, **params)

my_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False)

#my_model.fit(input=, output=,epochs=,batch_size=)
