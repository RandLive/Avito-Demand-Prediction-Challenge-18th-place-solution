#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:02:21 2018

@author: qifeng
"""


# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
import operator
from PIL import Image as IMG
from tqdm import tqdm




# In[7]:


def bin_spatial(img, size = (32,32)):
    path = img
    img1 = cv2.imread(path)
    img2 = cv2.resize(img1, size)
    return img2

def cs_conversion(img, cspace=['HSV', 'LUV', 'HLS', 'YUV']):
        feature_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        feature_LUV = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        feature_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        feature_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return feature_HSV, feature_LUV, feature_HLS, feature_YUV

def get_average_color(img):
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


def get_features(img_names):

    average_HSV_Hs = []
    average_HSV_Ss = []
    average_HSV_Vs = []
    average_LUV_Ls = []
    average_LUV_Us = []
    average_LUV_Vs = []
    average_HLS_Hs = []
    average_HLS_Ls = []
    average_HLS_Ss = []
    average_YUV_Ys = []
    average_YUV_Us = []
    average_YUV_Vs = []
    

    ids = []

    for img_name in tqdm(img_names):
        img_path = image_dir + img_name
        try:
            id_ = img_name[:-4]
            img = cv2.imread(img_path)
#            img = bin_spatial(img_path, size=(32,32))
            feature_HSV, feature_LUV, feature_HLS, feature_YUV = cs_conversion(img, cspace=['HSV', 'LUV', 'HLS', 'YUV'])
            
            average_HSV = get_average_color(feature_HSV)
            average_LUV = get_average_color(feature_LUV)
            average_HLS = get_average_color(feature_HLS)
            average_YUV = get_average_color(feature_YUV)
            
            average_HSV_H = average_HSV[0]/ 255.
            average_HSV_S = average_HSV[1]/ 255.
            average_HSV_V = average_HSV[2]/ 255.
            average_LUV_L = average_LUV[0]/ 255.
            average_LUV_U = average_LUV[1]/ 255.
            average_LUV_V = average_LUV[2]/ 255.
            average_HLS_H = average_HLS[0]/ 255.
            average_HLS_L = average_HLS[1]/ 255.
            average_HLS_S = average_HLS[2]/ 255.
            average_YUV_Y = average_YUV[0]/ 255.
            average_YUV_U = average_YUV[1]/ 255.
            average_YUV_V = average_YUV[2]/ 255.
            
 
        except:
            continue
        
        average_HSV_Hs.append(average_HSV_H)
        average_HSV_Ss.append(average_HSV_S)
        average_HSV_Vs.append(average_HSV_V)
        average_LUV_Ls.append(average_LUV_L)
        average_LUV_Us.append(average_LUV_U)
        average_LUV_Vs.append(average_LUV_V)
        average_HLS_Hs.append(average_HLS_H)
        average_HLS_Ls.append(average_HLS_L)
        average_HLS_Ss.append(average_HLS_S)
        average_YUV_Ys.append(average_YUV_Y)
        average_YUV_Us.append(average_YUV_U)
        average_YUV_Vs.append(average_YUV_V)
        ids.append(id_)
      
    return ids, average_HSV_Hs, average_HSV_Ss, average_HSV_Vs, average_LUV_Ls, average_LUV_Us, average_LUV_Vs, average_HLS_Hs, average_HLS_Ls, average_HLS_Ss, average_YUV_Ys, average_YUV_Us, average_YUV_Vs
 
# def paralleize(img_names):
#     img_names_split = np.array_split(img_names)
#     pool = Pool(20)
#     feature_list = pool.map(get_features, img_names_split)
#     for feature_index in range(len(feature_list[0])):
#         np.[feature_list[list_index][feature_index] for list_index in range(len(feature_list))]
            
image_dir = 'input/train_jpg/data/competition_files/train_jpg/'
print('getting train img names')               

# In[5]:

img_names = [i for i in os.walk(image_dir)][0][2]

'''
for subdir, dirs, files in os.walk(image_dir):
    for file in files:
        img_names = os.path.join(subdir, file)
'''
print(f'size of images: {len(img_names)}')

ids, average_HSV_Hs, average_HSV_Ss, average_HSV_Vs, average_LUV_Ls, average_LUV_Us, average_LUV_Vs, average_HLS_Hs, average_HLS_Ls, average_HLS_Ss, average_YUV_Ys, average_YUV_Us, average_YUV_Vs = get_features(img_names)

    # print(feature.shape)
cspace = {}

cspace['average_HSV_Hs'] = average_HSV_Hs
cspace['average_HSV_Ss'] = average_HSV_Ss
cspace['average_HSV_Vs'] = average_HSV_Vs
cspace['average_LUV_Ls'] = average_LUV_Ls
cspace['average_LUV_Us'] = average_LUV_Us
cspace['average_LUV_Vs'] = average_LUV_Vs
cspace['average_HLS_Hs'] = average_HLS_Hs
cspace['average_HLS_Ls'] = average_HLS_Ls
cspace['average_HLS_Ss'] = average_HLS_Ss
cspace['average_YUV_Ys'] = average_YUV_Ys
cspace['average_YUV_Us'] = average_YUV_Us
cspace['average_YUV_Vs'] = average_YUV_Vs
cspace['ids'] = ids

with open('input/train_image_features_cspace.p','wb') as f:
    pickle.dump(cspace, f)  
del cspace
del ids,\
    average_HSV_Hs,\
    average_HSV_Ss,\
    average_HSV_Vs,\
    average_LUV_Ls,\
    average_LUV_Us,\
    average_LUV_Vs,\
    average_HLS_Hs,\
    average_HLS_Ls,\
    average_HLS_Ss,\
    average_YUV_Ys,\
    average_YUV_Us,\
    average_YUV_Vs
del img_names
import gc
gc.collect()