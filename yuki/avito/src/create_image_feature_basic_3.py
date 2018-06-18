## http://maths-people.anu.edu.au/~johnm/courses/mathdm/talks/dimitri-clickadvert.pdf
## Grobal Features
## Brightness, Saturation, Colorfulness, Naturalness, Contrast, Sharpness,
## Texture, Grayscale simplicity, RGB simplicity, Color harmony, Hue histogram
## Local Features
## Basic segment statistics, Segment hue histogram, Segment color harmony, Segment brightness
## High level FEatures
## Interest points, Saliency map, Text, Human faces

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import gzip
import gc
from scipy.stats import skew, kurtosis, entropy
from joblib import Parallel, delayed
from scipy.ndimage import sobel
from utils import *
import pytesseract
import warnings
from saliency import Saliency
warnings.filterwarnings("ignore")

# train_file_dir = '../input/train_jpg'
# test_file_dir = '../input/test_jpg'
train_file_dir = '../input/data/competition_files/train_jpg'
test_file_dir = '../input/data/competition_files/test_jpg'


from PIL import Image
import cv2

def getstats(arr):
    ave = np.mean(arr)
    std = np.std(arr)
    ske = skew(arr.ravel())
    kur = kurtosis(arr.ravel())
    ent = entropy(arr.ravel())
    return [ave, std, ske, kur, ent]


def get_interest_points(arr):
    size = arr.shape[0] * arr.shape[1] / 100
    gray= cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    return [len(kp), len(kp)/size]


def get_saliency_map(arr):
    size = arr.shape[0] * arr.shape[1]
    sali = Saliency(arr)
    m = sali.get_saliency_map()
    sali_stats = getstats(m)
    cnt = (m>0.80).sum()
    cnt2 = (m>0.9).sum()
    cnt3 = (m<0.1).sum()
    cnt4 = (m<0.2).sum()
    return sali_stats + [cnt, cnt/size, cnt2, cnt2/size, cnt3, cnt3/size, cnt4, cnt4/size]


def get_data_from_image(image_path, f):
    try:
        cv_img = cv2.imread(image_path)
        bw = cv2.imread(image_path,0)
        yuv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2YUV)
        hls_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HLS)
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2HSV)
        img_size_x,img_size_y = cv_img.shape[0], cv_img.shape[1]
        img_size = img_size_x * img_size_y
        pixels = cv_img.shape[0] * cv_img.shape[1] * cv_img.shape[2]
    except:
        return [0] * num_feat

    #Saliency map,Interest points
    interest_points = get_interest_points(cv_img)
    saliency_map = get_saliency_map(cv_img)
    output = interest_points + saliency_map
    return output


def get_features(f, filedir):
    if f=="NAN":
        return [np.nan] * num_feat
    else:
        image_name = os.path.join(filedir, f+'.jpg')
        return get_data_from_image(image_name, f)

cols = ["interest_points_1", "interest_points_2"] + ["saliency_map_{}".format(i+1) for i in range(13)]

num_feat = len(cols)
print("NUM Features", num_feat)
print("train data...")
train_image_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = train_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in train_image_ids])
df_out = pd.DataFrame(out, columns=cols)

del out,train_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_3_features_train.parquet")
del df_out; gc.collect()

print("test data...")
test_image_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = test_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in test_image_ids])
df_out = pd.DataFrame(out, columns=cols)
del out, test_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_3_features_test.parquet")
