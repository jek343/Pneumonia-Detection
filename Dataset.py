#!/usr/bin/env python
# coding: utf-8

# In[5]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import data_manipulation
import pickle
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.exposure import adjust_sigmoid
from skimage import feature
from skimage.color import rgb2gray


# In[4]:


#data_manipulation.datset_split()


# In[6]:


with open("data/dataset_train.obj", "rb") as f:
    train = pickle.load(f)


# # Healthy Lung

# In[7]:


mask, class_ids = train.load_mask(0)
image = train.load_image(0)
plt.imshow(image)
plt.show()


# In[8]:


threshold_image = adjust_sigmoid(image)
plt.imshow(threshold_image)
plt.show()


# In[9]:


edge_detected = feature.canny(rgb2gray(threshold_image), sigma = 0.5)
plt.imshow(edge_detected)
plt.show()


# # Lung Opacity

# In[10]:


image_id = 15
mask, class_ids = train.load_mask(image_id)
image = train.load_image(image_id)
plt.imshow(image)
plt.show()


# In[11]:


threshold_image = adjust_sigmoid(image)
plt.imshow(threshold_image)
plt.show()


# In[12]:


masked = np.zeros(image.shape[:2])
for i in range(mask.shape[2]):
    masked += image[:, :, 0] * mask[:, :, i]
plt.imshow(masked, cmap='gray')


# In[13]:


edge_detected = feature.canny(rgb2gray(threshold_image), sigma = 0.6)
plt.imshow(edge_detected)
plt.show()


# In[14]:


def prior():
    ids = range(train.size())
    #sample = random.sample(ids, 20)
    cnt_box = np.zeros([1024,1024,1])

    p_imgs = [] #build up list of masks (numpy array) of images with pneumonia
    hasPneu = 0
    for id in ids[:]:
        mask, class_ids = train.load_mask(id) #mask = numpy array of boolean values. 1 if pneumonia in that pixel
                   #class_ids = 1 elt array. 1 if image has pneumonia. 0 otherwise
        if (class_ids.all()):
            hasPneu += 1
            #np.sum
            cnt_box = cnt_box + np.sum(mask.astype(int), axis = 2)
    #print(cnt_box.shape)
    cnt_box = cnt_box / hasPneu
    #x = np.max(cnt_box)
    return cnt_box


# In[15]:

if __name__ == "__main__":
    cnt_box = prior()
    np.save("Prior", cnt_box)

