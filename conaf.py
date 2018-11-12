# conaf.py
# 21st Oct. 2018
# Cornell Data Science

import data_manipulation
import data_generator

import keras as k
from keras import initializers
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Activation, UpSampling2D
from keras.models import Model
import numpy as np
import tensorflow as tf

# ========== CONSTANTS ========== START
IMAGE_DIM = 128

# ========== MODEL PARAMS ========== START
PARAMS = {'dim': (IMAGE_DIM,IMAGE_DIM),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': True}

CLASS_OUTPUT_NAME = "output1"
LOC_OUTPUT_NAME = "output2"

LOSSES = {CLASS_OUTPUT_NAME : "categorical_crossentropy",
          LOC_OUTPUT_NAME : "mean_squared_error"}
LOSS_WEIGHTS = {CLASS_OUTPUT_NAME : 1.0,
                LOC_OUTPUT_NAME : 1.0}

def create_localizer_branch(in_layer):
    local_conv2d_1 = Conv2D(filters = 256, kernel_size = (1,1), padding = 'same', name = 'localizer_conv2d_1')(in_layer)
    local_conv2d_2 = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', name = 'localizer_conv2d_2')(local_conv2d_1)
    local_conv2d_3 = Conv2D(filters = 32, kernel_size = (1,1), padding = 'same', name = 'localizer_conv2d_3')(local_conv2d_2)
    local_conv2d_4 = Conv2D(filters = 1, kernel_size = (1,1), padding = 'same', name = 'localizer_conv2d_4')(local_conv2d_3)
    sigmoid_activation1 = Activation('sigmoid', name = 'output2')(local_conv2d_4)
    #Assuming IMAGE DIM is 1024 --> create 7 upsampling layers
    up_conv1 = 
    return sigmoid_activation1

def create_classifier_branch(in_layer):
    class_maxpool1 = MaxPool2D(pool_size=(2,2), strides = None, padding='same', name = 'classifier_maxpool1')(in_layer)
    class_conv2d_1 = Conv2D(filters = 256, kernel_size=(1,1), padding='same', name='classifier_conv2d_1',
                        activation='relu')(class_maxpool1)
    class_conv2d_2 = Conv2D(filters = 56, kernel_size=(1,1), padding='same', name='classifier_conv2d_2',
                        activation='relu')(class_conv2d_1)
    class_conv2d_3 = Conv2D(filters = 28, kernel_size=(1,1), padding='same', name='classifier_conv2d_3',
                        activation='relu')(class_conv2d_2)

    flatten_1 = Flatten()(class_conv2d_3)
    class_dense_1 = Dense(units = 2, activation ='softmax', name = 'output1')(flatten_1)

    return class_dense_1

def create_model():
    res50 = ResNet50(weights='imagenet', include_top=False, input_shape = (IMAGE_DIM, IMAGE_DIM, 3))

    class_branch_output = create_classifier_branch(res50.layers[-40].output)
    loc_branch_output = create_localizer_branch(res50.layers[-40].output)

    model = Model(input=res50.input, output = [class_branch_output, loc_branch_output])
    return model

''' train_set: DetectorDataset object for the training set.
    val_set  : DetectorDataset object for the validation set. '''
def train_model(model, train_set, val_set):
    model = create_model()
    model.compile(optimizer='adam', loss=LOSSES, loss_weights=LOSS_WEIGHTS, metrics=['accuracy'])

    train_labels = train.labels_list()
    val_labels = val.labels_list()

    training_generator = data_generator.DataGenerator(train, range(train.size()), train_labels, **PARAMS)
    validation_generator = data_generator.DataGenerator(val, range(val.size()), val_labels, **PARAMS)

    #TODO fit generator (ensure generator returns two outputs)
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False)

def main():
    m = create_model()
    print(m.summary())


if __name__ == "__main__":
    main()

