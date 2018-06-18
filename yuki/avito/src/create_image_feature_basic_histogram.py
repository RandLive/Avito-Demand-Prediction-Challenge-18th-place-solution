import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import gzip
import gc
from keras.preprocessing import image
from scipy.stats import skew, kurtosis, entropy
from joblib import Parallel, delayed
from utils import *

train_file_dir = '../input/train_jpg'
test_file_dir = '../input/test_jpg'

bins = 255

from PIL import Image
import cv2

def get_hist_from_image(image_path, f):
    try:
        cv_img = cv2.imread(image_path)
        img_size = [cv_img.shape[0], cv_img.shape[1]]
    except:
        return [0] * bins

    return list(np.histogram(cv_img, bins=bins, range=(0, 255))[0])


def get_features(f, filedir):
    if f=="NAN":
        return [0] * bins
    else:
        image_name = os.path.join(filedir, f+'.jpg')
        return get_hist_from_image(image_name, f)

y = pd.read_csv("../input/train.csv", usecols=["deal_probability"])["deal_probability"].values

print("train data...")
train_image_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = train_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in train_image_ids])
X_train = np.array(out)
del out,train_image_ids; gc.collect()

print("test data...")
test_image_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = test_file_dir
out = Parallel(n_jobs=-1, verbose=1)([delayed(get_features)(f,file_dir) for f in test_image_ids])
X_test = np.array(out)
del out, test_image_ids; gc.collect()

# oof_sgd(X_train,X_test,y,"img_histgram")
# oof_lgbm(X_train,X_test,y,"img_histgram")
