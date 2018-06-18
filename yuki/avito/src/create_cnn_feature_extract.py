import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Concatenate, SpatialDropout1D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
from keras.engine import InputSpec, Layer
from tqdm import tqdm
import sys, os, re, csv, codecs, time, gc, argparse, logging, traceback
import dill
from utils import *

train_file_dir = '../input/train_jpg'
test_file_dir = '../input/test_jpg'
batch_size = 128
num_epoch = 100
patience = 2
image_x = 224
image_y = 224
weight_path = "cnn.h5"
n_train = pd.read_csv("../input/train.csv").shape[0]
n_test = pd.read_csv("../input/test.csv").shape[0]

def get_id_chunks(ix_train, batch_size):
    np.random.shuffle(ix_train)
    length = int(len(ix_train) / batch_size) + 1
    for i in range(length):
        yield ix_train[i*batch_size:(i+1)*batch_size]


def resize_img(im, inter=cv2.INTER_AREA):
    height, width, _ = im.shape
    if height > width:
        new_dim = (width*//height, im_dim)
    else:
        new_dim = (im_dim, height*im_dim//width)

    imr = cv2.resize(im, new_dim, interpolation=inter)
    h, w = imr.shape[:2]
    off_x = (im_dim-w)//2
    off_y = (im_dim-h)//2
    im_out = np.zeros((im_dim, im_dim, n_channels), dtype=imr.dtype)
    im_out[off_y:off_y+h, off_x:off_x+w] = imr
    del imr
    return im_out


def get_image(ids, train_or_test="train"):
    if train_or_test=="train":
        file_dir = train_file_dir
    elif train_or_test=="test":
        file_dir = test_file_dir
    out = np.zeros((len(ids), image_x, image_y, 3))
    for i in range(len(ids)):
        try:
            img = cv.imread(os.path.join(file_dir, i+".jpg"))
            img = resize_img(img)
        except:
            img = np.zeros((image_x, image_y, 3))
        out[i, :, :, :] = img
    return out

def rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power((y_true - y_pred), 2)))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true-y_pred)))


model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(image_x, image_y, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))
model.compile(loss=root_mean_squared_error,
              optimizer="rmsprop",
              metrics=[root_mean_squared_error])


image_ids = pd.read_csv("../input/train.csv",usecols=["image_id"]).image_id.values
image_ids_test = pd.read_csv("../input/test.csv",usecols=["image_id"]).image_id.values
nsplits = 5
with open("../tmp/oof_index.dat", "rb") as f:
    kfolds = dill.load(f)
val_score = []
result = np.zeros((n_test, 1))
cnt = 0
oof_valid = np.zeros((n_train, 1))

for ix_train, ix_valid in kfolds:
    print("=======CV=======")
    val_ids = image_ids[ix_valid]
    X_val = get_image(val_ids, "train")
    y_val = y[ix_valid]
    bst_val_score = 1
    early_stop = 0
    for epoch in range(num_epoch):
        print("======={}epoch=======".format(epoch))
        for ixs in tqdm(get_id_chunks(ix_train, batch_size), total=int(ix_train/batch_size)+1):
            train_ids = image_ids[ixs]
            X_train = get_image(train_ids, "train")
            y_train = y[ixs]
            model.fit(X
                    , y_train
                    , batch_size=batch_size
                    , epoch=1
                    , verbose=1
                    , shuffle=True
                    # , validation_data=(X_val, y_val)
                    )
        val_score = rmse(y_val, model.predict(X_val))
        if val_score < bet_val_score:
            bst_val_score = val_score
            model.save_weights(weight_path)
            early_stop = 0
        else:
            early_stop += 1
        if early_stop > patience:
            print("Early Stopping!! Best Epoch {}".format(epoch))
            model.load_weights(weight_path)
            break

    for i in range(int(len(image_ids_test)/batch_size)+1):
        test_ids = image_ids[i*batch_size:(i+1)*batch_size]
        X_test = get_image(test_ids, "test")
        result[i*batch_size:(i+1)*batch_size] += model.predict(X_test) / nsplits
    oof_valid[ix_valid, :] = model.predict(X_val)
    K.clear_session()
    gc.collect()

df_out = pd.DataFrame(result, columns=["oof_cnn_image_feature"])
to_parquet(df_out, "../features/oof_cnn_image_feature_test.parquet")

df_out = pd.DataFrame(oof_valid, columns=["oof_cnn_image_feature"])
to_parquet(df_out, "../features/oof_cnn_image_feature_train.parquet")
