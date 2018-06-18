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
warnings.filterwarnings("ignore")

# train_file_dir = '../input/train_jpg'
# test_file_dir = '../input/test_jpg'
train_file_dir = '../input/data/competition_files/train_jpg'
test_file_dir = '../input/data/competition_files/test_jpg'

scl_min, scl_max = 75, 180
n_bins = 30

from PIL import Image
import cv2

def getstats(arr):
    ave = np.mean(arr)
    std = np.std(arr)
    ske = skew(arr)
    kur = kurtosis(arr)
    ent = entropy(arr)
    maximum = np.max(arr)
    minimum = np.min(arr)
    return [ave, std, ske, kur, ent, maximum, minimum]

def get_colorfulness(arr):
    (B, G, R) = cv2.split(arr.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

def get_contrast(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr = (arr - arr_min) / (arr_max - arr_min)
    return np.std(arr)

def get_num_above_thres(hist):
    return (hist > hist.mean() + 2* hist.std()).sum()


def get_gray_simplicity(bw):
    hist,_ = np.histogram(bw.ravel(),255,[0,255])
    thres = bw.shape[0] * bw.shape[1] * 0.05
    feat1 = 0
    for i in range(127):
        under = (hist<=i).sum()
        upper = (hist>=(255-i)).sum()
        if under+upper >= thres:
            feat1 = 255 - 2*i
            break
    feat2 = np.std(hist)
    feat3 = get_num_above_thres(hist)
    return [feat1, feat2, feat3]

def get_rgb_simplicity(arr):
    bins = 512
    pixels = arr.shape[0] * arr.shape[1] * arr.shape[2]
    r,_ = np.histogram(arr[:, :, 0].ravel(),bins,[0,bins])
    g,_ = np.histogram(arr[:, :, 1].ravel(),bins,[0,bins])
    b,_ = np.histogram(arr[:, :, 2].ravel(),bins,[0,bins])
    feat_r1 = get_num_above_thres(r)
    feat_g1 = get_num_above_thres(g)
    feat_b1 = get_num_above_thres(b)
    feat_r2 = np.max(r) / pixels
    feat_g2 = np.max(g) / pixels
    feat_b2 = np.max(b) / pixels
    return [feat_r1, feat_g1, feat_b1, feat_r2, feat_g2, feat_b2]

def get_hue_features(arr):
    bins = 20
    hist,_ = np.histogram(arr[:, :, 0].ravel(),bins,[0,bins])
    feat1 = get_num_above_thres(hist)
    # arclength? hue
    _,thresh = cv2.threshold(arr[:, :, 0][0],10,20,0)
    _,contours,_ = cv2.findContours(thresh, 1, 2)
    peris = [cv2.arcLength(cnt,True) for cnt in contours]
    return [feat1, np.std(peris)]

def face_detector(cv_img):
    pixels = cv_img.shape[0] * cv_img.shape[1] * cv_img.shape[2]
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(cv_img, cv_img, mask = skinMask)
    return (skin>0).sum()/pixels

def get_string(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    result = pytesseract.image_to_string(Image.fromarray(img))
    return [len(result), len(result.split())]


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

    # Grobal Features
    brightness = getstats(yuv_img[:, :, 0][0])
    harmony = getstats(hls_img[:, :, 0][0])
    lightness = getstats(hls_img[:, :, 0][1])
    saturation = getstats(hls_img[:, :, 0][2])
    colorfulness = get_colorfulness(cv_img)
    contrast = get_contrast(hls_img[:, :, 0][1])
    grayscale_simplicity = get_gray_simplicity(bw)
    rgb_simplicity = get_rgb_simplicity(cv_img)
    hsv_simplicity = get_rgb_simplicity(hsv_img)
    hue = get_hue_features(hsv_img)

    # High level Features
    face_ratio = face_detector(cv_img)
    charactor_feat = get_string(bw)
    #Saliency map,Interest points

    output = brightness + lightness + harmony + saturation + \
    [colorfulness, contrast] + grayscale_simplicity + rgb_simplicity + hsv_simplicity + hue + \
    [face_ratio] + charactor_feat
    return output


def get_features(f, filedir):
    if f=="NAN":
        return [np.nan] * num_feat
    else:
        image_name = os.path.join(filedir, f+'.jpg')
        return get_data_from_image(image_name, f)

stats_col = ["ave", "std", "skew", "kur", "entropy", "max", "min"]
cols = ["brightness_"+col for col in stats_col]+["lightness_"+col for col in stats_col]+\
["harmony_"+col for col in stats_col]+["saturation_"+col for col in stats_col]+\
["colorfulness", "contrast"] + ["grascale_simplicity_{}".format(i+1) for i in range(3)]+\
["rgb_simplicity_{}".format(i+1) for i in range(6)]+["hsv_simplicity_{}".format(i+1) for i in range(6)]+\
["hue_1", "hue_2"] + ["face_ratio"]+["charactor_feat_1","charactor_feat_2"]


num_feat = len(cols)
print("NUM Features", num_feat)
print("train data...")
train_image_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = train_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in train_image_ids])
df_out = pd.DataFrame(out, columns=cols)

del out,train_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_2_features_train.parquet")
del df_out; gc.collect()

print("test data...")
test_image_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = test_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in test_image_ids])
df_out = pd.DataFrame(out, columns=cols)
del out, test_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_2_features_test.parquet")
