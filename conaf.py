import numpy as np
import tensorflow as tf
import keras as k
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, MaxPool2D, Conv2D
from keras.models import Model
from keras import initializers

vgg16 = VGG16(weights='imagenet', include_top=False)

class_maxpool1 = MaxPool2D(pool_size=(2,2), strides = None, padding='same', name = 'classifier_maxpool1')(vgg16.layers[-1].output)
class_conv2d_1 = Conv2D(filters = 256, kernel_size=(1,1), padding='same', name='classifier_conv2d_1', 
                        activation='relu')(class_maxpool1)
class_conv2d_2 = Conv2D(filters = 56, kernel_size=(1,1), padding='same', name='classifier_conv2d_2', 
                        activation='relu')(class_conv2d_1)
class_conv2d_3 = Conv2D(filters = 28, kernel_size=(1,1), padding='same', name='classifier_conv2d_3', 
                        activation='relu')(class_conv2d_2)
class_dense_1 = Dense(units = 2, activation ='softmax')(class_conv2d_3)


my_model = Model(input=vgg16.input, output = class_dense_1)

print(my_model.summary())

my_model.compile(loss='categorical_crossentropy', optimizer='adam')

#my_model.fit(input=, output=,epochs=,batch_size=)