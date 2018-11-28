#!/usr/bin/env python
# coding: utf-8

import pickle
#mport pydicom
#import matplotlib.pyplot as plt
import numpy as np
#import random
#import csv
import os
#from skimage.exposure import adjust_sigmoid
#from skimage import feature
#from skimage.color import rgb2gray


with open("data/dataset_train.obj", "rb") as f:
    train = pickle.load(f)


IMAGE_DIM = 1024
SCALE = 1.2

def probability_distribution(image_id, width, height, scale, x, y):
    #maximum_distance is the maximum distance from the edge of the bounding box for the probability to be calculated
    #width is the width of the bounding box
    #height is the height of the bounding box
    #x is the xth pixel where the center of the bounding box is
    #y is the yth pixel where the center of the bounding box is
    #bounding boxes defined by x-min y-min???
    center_x = int(x+width/2)
    center_y = int(y+height/2)
    max_radius = int(np.sqrt((height/2)**2 + (width/2)**2))

    #assume cv2 used to load image
    x_dim = IMAGE_DIM
    y_dim = IMAGE_DIM
    new_image = np.zeros(shape=(x_dim,y_dim))
    
    for xi in range (int(-scale*width/2), int(scale*width/2)):
        for yi in range (int(-scale*height/2), int(scale*height/2)):
            r = np.sqrt((xi)**2 + (yi)**2)
            probability = 1.0-(r/(scale*max_radius))
            probability = 0 if probability < 0 else probability
            x_new_image = (center_x + xi)
            y_new_image = (center_y + yi)
            if(x_new_image > -1 and x_new_image < x_dim and y_new_image > -1 and y_new_image < y_dim ):
                new_image[x_new_image][y_new_image] = round(probability, 2) 
    
    return new_image
                
    


def getProbDist(image_id, scale):
    info = train.image_info[image_id]
    x_dim = IMAGE_DIM
    y_dim = IMAGE_DIM
    probDist = np.zeros(shape=(x_dim,y_dim))
    annotations = info['annotations']
    if(len(annotations) > 0):
        for a in annotations:
            if a['Target'] == 1:
                x = int(a['x'])
                y = int(a['y'])
                w = int(a['width'])
                h = int(a['height'])
                probDist += probability_distribution(image_id, h, w, scale, y, x)
    else:
        return np.zeros((IMAGE_DIM, IMAGE_DIM))
    return (np.clip(probDist, 0, 1))



def getProbFiles():
    path = os.getcwd() + '/data/ProbDistributions'
    if not os.path.exists(path):
        os.mkdir(path)
    
    for id in range(train.size()):
        patientId = train.image_info[id]['annotations'][0].patientId
        np.save(path + '/Distr' + str(patientId), getProbDist(id, SCALE))


if __name__ == "__main__":
    getProbFiles()




