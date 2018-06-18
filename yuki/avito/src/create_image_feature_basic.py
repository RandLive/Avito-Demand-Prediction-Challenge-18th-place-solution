import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import gzip
import gc
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
from scipy.stats import skew, kurtosis, entropy
from joblib import Parallel, delayed
from scipy.ndimage import sobel
from utils import *

train_file_dir = '../input/train_jpg'
test_file_dir = '../input/test_jpg'
scl_min, scl_max = 75, 180
n_bins = 30

from PIL import Image
import cv2

def getblur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_data_from_image(image_path, f):

    try:
        cv_img = cv2.imread(image_path)
        bw = cv2.imread(image_path,0)
        img_size = [cv_img.shape[0], cv_img.shape[1]]
    except:
        return [0] * (num_feat)
    (means, stds) = cv2.meanStdDev(cv_img)
    sum_color = np.sum(cv_img.flatten())
    mean_color = np.mean(cv_img.flatten())
    std_color = np.std(cv_img.flatten())
    skew_color = skew(cv_img.flatten())
    kur_color = kurtosis(cv_img.flatten())
    entropy_color = entropy(cv_img.flatten())

    red_sum_color = np.sum(cv_img[:,:,0].flatten())
    red_skew_color = skew(cv_img[:,:,0].flatten())
    red_kur_color = kurtosis(cv_img[:,:,0].flatten())
    red_entropy_color = entropy(cv_img[:,:,0].flatten())

    blue_sum_color = np.sum(cv_img[:,:,1].flatten())
    blue_skew_color = skew(cv_img[:,:,1].flatten())
    blue_kur_color = kurtosis(cv_img[:,:,1].flatten())
    blue_entropy_color = entropy(cv_img[:,:,1].flatten())

    green_sum_color = np.sum(cv_img[:,:,2].flatten())
    green_skew_color = skew(cv_img[:,:,2].flatten())
    green_kur_color = kurtosis(cv_img[:,:,2].flatten())
    green_entropy_color = entropy(cv_img[:,:,2].flatten())
    moments = [red_sum_color, red_skew_color, red_kur_color, red_entropy_color,\
            blue_sum_color, blue_skew_color, blue_kur_color, blue_entropy_color, \
            green_sum_color, green_skew_color, green_kur_color, green_entropy_color]

    color_stats = np.concatenate([means, stds]).flatten()
    blur = getblur(cv_img)
    # hist = list(cv2.calcHist([bw],[0],None,[64],[0,64]).flatten())
    bw_under_10 = (bw<10).sum()
    bw_over_245 = (bw>245).sum()
    # sobel
    sobels = []
    for i in range(3):
        sobel0 = sobel(cv_img[:, :, i], axis=0, mode='reflect', cval=0.0).ravel().var()
        sobel1 = sobel(cv_img[:, :, i], axis=1, mode='reflect', cval=0.0).ravel().var()
        sobels += [sobel0, sobel1]
    # hists = list(np.histogram(cv_img, bins=n_bins, range=(scl_min, scl_max))[0])
    output = img_size  + [sum_color, mean_color, std_color, skew_color, kur_color, entropy_color]\
     + color_stats.tolist() + [blur] + moments + [bw_under_10, bw_over_245] + sobels

    return output


def get_features(f, filedir):
    if f=="NAN":
        return [np.nan] * num_feat
    else:
        image_name = os.path.join(filedir, f+'.jpg')
        return get_data_from_image(image_name, f)


cols = ['img_size_x', 'img_size_y', 'img_sum_color', 'img_mean_color', 'img_std_color', 'img_skew_color', 'img_kur_color', 'img_entropy_color', \
     'img_blue_mean', 'img_green_mean', 'img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std',\
     'img_blur', \
     'img_red_sum_color','img_red_skew_color','img_red_kur_color','img_red_entropy_color', \
     'img_blue_sum_color', 'img_blue_skew_color','img_blue_kur_color','img_blue_entropy_color',\
     'img_green_sum_color','img_green_skew_color','img_green_kur_color','img_green_entropy_color',\
     'img_bw_under_10', 'img_bw_over_245'] + \
     ["img_sobel_{}_{}".format(i+1, j+1) for i in range(3) for j in range(2)]

num_feat = len(cols)
print("NUM Features", num_feat)
print("train data...")
train_image_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = train_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in train_image_ids])
df_out = pd.DataFrame(out, columns=cols)
del out,train_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_features_train.parquet")
del df_out; gc.collect()

print("test data...")
test_image_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = test_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in test_image_ids])
df_out = pd.DataFrame(out, columns=cols)
del out, test_image_ids; gc.collect()
to_parquet(df_out, "../features/fe_img_basic_features_test.parquet")
