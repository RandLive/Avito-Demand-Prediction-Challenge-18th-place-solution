
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


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

# In[7]:
def perform_color_analysis(img, flag):
    path = img
    im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

def average_pixel_width_(img):
    path = img
    im = IMG.open(path)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(img):
    path = img
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color

def get_average_color(img):
    path = img
    img = cv2.imread(path)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

def getSize(filename):
    filename = filename
    st = os.stat(filename)
    return st.st_size

def getDimensions(filename):
    filename = filename
    img_size = IMG.open(filename).size
    return img_size 

def get_blurrness_score(image):
    path =  image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def get_features(img_names):
    dullnesses = []
    whitenesses = []
    dominant_reds = []
    average_pixel_widths = []
    dominant_greens = []
    dominant_blues = []
    average_reds = []
    average_greens = []
    average_blues = []
    image_sizes = []
    widths = []
    heights = []
    blurinesses = []
    ids = []

    for img_name in img_names:
        img_path = image_dir + img_name
        try:
            id_ = img_name[:-4]
            dullness = perform_color_analysis(img_path, 'black')
            whiteness = perform_color_analysis(img_path, 'white')
            average_pixel_width = average_pixel_width_(img_path)
            dominant_color = get_dominant_color(img_path)
            dominant_red = dominant_color[0] / 255.
            dominant_green = dominant_color[1] / 255.
            dominant_blue = dominant_color[2] / 255.    
            average_color = get_average_color(img_path)
            average_red = average_color[0] / 255.
            average_green = average_color[1] / 255.
            average_blue = average_color[2] / 255.  
            image_size = getSize(img_path)
            width, height = getDimensions(img_path)
            bluriness = get_blurrness_score(img_path)
        except OSError:
            return   [None] * 14
        dullnesses.append(dullness)
        whitenesses.append(whiteness)
        average_pixel_widths.append(average_pixel_width)
        dominant_reds.append(dominant_red)
        dominant_greens.append(dominant_green)
        dominant_blues.append(dominant_blue)
        average_reds.append(average_red)
        average_greens.append(average_green)
        average_blues.append(average_blue)
        image_sizes.append(image_size)
        widths.append(width)
        heights.append(height)
        blurinesses.append(bluriness)
        ids.append(id_)
        
    return ids, dullnesses, whitenesses, average_pixel_widths, dominant_reds, dominant_greens, dominant_blues, average_reds, average_greens, average_blues, image_sizes, widths, heights, blurinesses


# def paralleize(img_names):
#     img_names_split = np.array_split(img_names)
#     pool = Pool(20)
#     feature_list = pool.map(get_features, img_names_split)
#     for feature_index in range(len(feature_list[0])):
#         np.[feature_list[list_index][feature_index] for list_index in range(len(feature_list))]
            
            
    
    
image_dir = '../input/test_jpg/data/competition_files/test_jpg/'

print('getting img names')
# In[5]:


img_names = [i for i in os.walk(image_dir)][0][2]
print(f'size of images: {len(img_names)}')

dullnesses = []
whitenesses = []
dominant_reds = []
average_pixel_widths = []
dominant_greens = []
dominant_blues = []
average_reds = []
average_greens = []
average_blues = []
image_sizes = []
widths = []
heights = []
blurinesses = []
ids = []

for img_name in tqdm(img_names):
    
    img_path = image_dir + img_name
    try:
        id_ = img_name[:-4]
        dullness = perform_color_analysis(img_path, 'black')
        whiteness = perform_color_analysis(img_path, 'white')
        average_pixel_width = average_pixel_width_(img_path)
#         dominant_color = get_dominant_color(img_path)
#         dominant_red = dominant_color[0] / 255.
#         dominant_green = dominant_color[1] / 255.
#         dominant_blue = dominant_color[2] / 255.    
#         average_color = get_average_color(img_path)
#         average_red = average_color[0] / 255.
#         average_green = average_color[1] / 255.
#         average_blue = average_color[2] / 255.  
#         image_size = getSize(img_path)
#         width, height = getDimensions(img_path)
        bluriness = get_blurrness_score(img_path)
    except OSError:
        continue    
    dullnesses.append(dullness)
    whitenesses.append(whiteness)
    average_pixel_widths.append(average_pixel_width)
#     dominant_reds.append(dominant_red)
#     dominant_greens.append(dominant_green)
#     dominant_blues.append(dominant_blue)
#     average_reds.append(average_red)
#     average_greens.append(average_green)
#     average_blues.append(average_blue)
#     image_sizes.append(image_size)
#     widths.append(width)
#     heights.append(height)
    blurinesses.append(bluriness)
    ids.append(id_)
    
    
    # print(feature.shape)

    

x = {}
x['dullnesses'] = dullnesses
x['whitenesses'] = whitenesses
x['average_pixel_width'] = average_pixel_width
x['dominant_reds'] = dominant_reds
x['dominant_blues'] = dominant_blues
x['dominant_greens'] = dominant_greens
x['average_reds'] = average_reds
x['average_blues'] = average_blues
x['average_greens'] = average_greens
x['image_sizes'] = image_sizes
x['widths'] = widths
x['heights'] = heights
x['blurinesses'] = blurinesses
x['ids'] = ids
with open('../input/test_image_features.p','wb') as f:
    pickle.dump(x, f)  
    

del x
del ids
del img_names
import gc
gc.collect()

image_dir = '../input/train_jpg/data/competition_files/train_jpg/'

print('getting img names')
# In[5]:


img_names = [i for i in os.walk(image_dir)][0][2]
print(f'size of images: {len(img_names)}')

dullnesses = []
whitenesses = []
dominant_reds = []
average_pixel_widths = []
dominant_greens = []
dominant_blues = []
average_reds = []
average_greens = []
average_blues = []
image_sizes = []
widths = []
heights = []
blurinesses = []
ids = []

for img_name in tqdm(img_names):
    
    img_path = image_dir + img_name
    try:
        id_ = img_name[:-4]
        dullness = perform_color_analysis(img_path, 'black')
        whiteness = perform_color_analysis(img_path, 'white')
        average_pixel_width = average_pixel_width_(img_path)
#         dominant_color = get_dominant_color(img_path)
#         dominant_red = dominant_color[0] / 255.
#         dominant_green = dominant_color[1] / 255.
#         dominant_blue = dominant_color[2] / 255.    
#         average_color = get_average_color(img_path)
#         average_red = average_color[0] / 255.
#         average_green = average_color[1] / 255.
#         average_blue = average_color[2] / 255.  
#         image_size = getSize(img_path)
#         width, height = getDimensions(img_path)
        bluriness = get_blurrness_score(img_path)
    except OSError:
        continue    
    dullnesses.append(dullness)
    whitenesses.append(whiteness)
    average_pixel_widths.append(average_pixel_width)
#     dominant_reds.append(dominant_red)
#     dominant_greens.append(dominant_green)
#     dominant_blues.append(dominant_blue)
#     average_reds.append(average_red)
#     average_greens.append(average_green)
#     average_blues.append(average_blue)
#     image_sizes.append(image_size)
#     widths.append(width)
#     heights.append(height)
    blurinesses.append(bluriness)
    ids.append(id_)
    
    
    # print(feature.shape)

    

x = {}
x['dullnesses'] = dullnesses
x['whitenesses'] = whitenesses
x['average_pixel_width'] = average_pixel_width
x['dominant_reds'] = dominant_reds
x['dominant_blues'] = dominant_blues
x['dominant_greens'] = dominant_greens
x['average_reds'] = average_reds
x['average_blues'] = average_blues
x['average_greens'] = average_greens
x['image_sizes'] = image_sizes
x['widths'] = widths
x['heights'] = heights
x['blurinesses'] = blurinesses
x['ids'] = ids
with open('../input/train_image_features.p','wb') as f:
    pickle.dump(x, f)  