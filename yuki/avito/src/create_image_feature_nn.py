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
from itertools import chain
from utils import *

train_file_dir = '../input/train_jpg'
test_file_dir = '../input/test_jpg'
im_dim = 224
batch_size = 128
n_channels = 3
empty_im = np.zeros((im_dim, im_dim, n_channels), dtype=np.uint8) # Used when no image is present
from PIL import Image
import cv2

print("train data...")
train_image_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = train_file_dir


print("test data...")
test_image_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
file_dir = test_file_dir


# NN features(confidence): GPU intensive
cols = ['Resnet50_label', 'Resnet50_score', 'xception_label', 'xception_score', 'Inception_label', 'Inception_score']
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')


def image_classify(model, pak, images, labelname,top_n=1):
    """Classify image and return top matches."""
    target_size = (im_dim, im_dim)
    tmp = []
    for img in images:
        if img=="NAN":
            x = empty_im
        else:
            if img.size != target_size:
                img = img.resize(target_size)
            x = image.img_to_array(img)
        tmp.append(x)
    x = np.array(tmp)
    del tmp, images, img; gc.collect()
    x = pak.preprocess_input(x)
    preds = model.predict(x, batch_size=batch_size)
    return np.array(list(chain.from_iterable(pak.decode_predictions(preds, top=1))))[:, 1:]



def classify(image_ids, file_dir):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    image_paths = [os.path.join(file_dir, i+".jpg") for i in image_ids]
    images = []
    for image_path in image_paths:
        try:
            images.append(Image.open(image_path))
        except:
            images.append("NAN")

    resnet_preds = image_classify(resnet_model, resnet50, images, "Resnet50")
    xception_preds = image_classify(xception_model, xception, images, "xception")
    inception_preds = image_classify(inception_model, inception_v3, images, "Inception")
    return np.concatenate([resnet_preds, xception_preds,inception_preds], axis=1)


# Train
df_train = pd.read_csv("../input/train.csv", chunksize=batch_size, usecols=["image"])
file_dir = train_file_dir
out = []
for df_chunk in tqdm(df_train, total=int(1503424/batch_size)):
    image_ids = df_chunk["image"].fillna("NAN").tolist()
    out.append(classify(image_ids, file_dir))

df_out = pd.DataFrame(np.concatenate(out), columns=cols)
train_ids = pd.read_csv("../input/train.csv", usecols=["image"])["image"].fillna("NAN").tolist()
df_out["image"] = train_ids
del out; gc.collect()
to_parquet(df_out, "../tmp/fe_img_pretrained_nnmodel_train.parquet")

# Test
df_test = pd.read_csv("../input/test.csv", chunksize=batch_size, usecols=["image"])
file_dir = test_file_dir
out = []

for df_chunk in tqdm(df_test, total=int(508438/batch_size)):
    image_ids = df_chunk["image"].fillna("NAN").tolist()
    out.append(classify(image_ids, file_dir))

df_out = pd.DataFrame(np.concatenate(out), columns=cols)
test_ids = pd.read_csv("../input/test.csv", usecols=["image"])["image"].fillna("NAN").tolist()
df_out["image"] = test_ids
del out; gc.collect()
to_parquet(df_out, "../tmp/fe_img_pretrained_nnmodel_test.parquet")
