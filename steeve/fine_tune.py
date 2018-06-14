import pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from ImageGenerator import *
from sklearn.model_selection import KFold
from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dropout, Dense, concatenate, CuDNNGRU, Embedding, Flatten, Activation, BatchNormalization, PReLU
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K
from tqdm import tqdm
from nltk import ngrams
from keras.backend.tensorflow_backend import set_session
from sklearn.metrics import mean_squared_error
import os
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

train_dir = '../input/train_jpg/data/competition_files/train_jpg/'
test_dir = '../input/test_jpg/data/competition_files/test_jpg/'

# restrict gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def get_model():
    #Load the VGG model
#     vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(160,160, 3))
    vgg_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(160,160, 3))
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False


    model = models.Sequential() 
    # Add the vgg convolutional base model
    model.add(vgg_conv)
    model.add(BatchNormalization())
    # Add new layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))    
    optimizer = optimizers.Adam(0.0001, amsgrad=True)

    model.compile(loss="mse", optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))
    return model


def train_bagging(X, y, fold_count):
    
    
    kf = KFold(n_splits=fold_count, random_state=42, shuffle=True)
#     skf = StratifiedKFold(n_splits=fold_count, random_state=None, shuffle=False)
    fold_id = -1
#     model_list = []
    val_predict= np.zeros(y.shape)
#     rmse_list = []
    for train_index, test_index in kf.split(y):
        
        fold_id +=1 
        if fold_id >= 1: exit()

        print(f'fold number: {fold_id}', flush=True)
        
        
#         x_train, x_val = X[train_index], X[test_index]
#         print(X.head())
#         print(X.index)
        
        x_train, x_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        x_train.set_index('item_id', inplace=True)
        x_val.set_index('item_id', inplace=True)        
        
        train_item_ids = x_train.index
        val_item_ids =  x_val.index
        
        train_image_ids = x_train.image
        val_image_ids = x_val.image
        
        train_labels = x_train.deal_probability
        val_labels = x_val.deal_probability
        
#         print(val_labels)
        train_gen = ImageGenerator(train_dir, train_item_ids, train_image_ids, train_labels)
        val_gen = ImageGenerator(train_dir, val_item_ids, val_image_ids, val_labels)
        model_path = f'../weights/{fname}_fold{fold_id}.hdf5'
        
        
        model = get_model()
        

        early= EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#         rlrop = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=2,verbose=1,factor=0.1,cooldown=0,min_lr=1e-6)
        callbacks = [early, checkpoint]
        
        model.fit_generator(train_gen,  validation_data=val_gen, callbacks=callbacks, epochs=epochs, verbose=1)
        model.load_weights(model_path)
        y_pred = model.predict(x_val)        
        val_predict[test_index] = y_pred[:,0]
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        train_rmse = mean_squared_error(model.predict(x_train), y_train) ** 0.5
        print(f'train_rmse {train_rmse}')
        print(f'rmse: {rmse}')
        y_pred = model.predict(x_test)
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['deal_probability'] = y_pred
        sub['deal_probability'].clip(0.0, 1.0, inplace=True)
        sub.to_csv(f'../output/{fname}_test_fold{fold_id}.csv', index=False)
        del model
        gc.collect()
        rmse_list.append(rmse)
#         model_list.append(model)
    print(f'rmse score avg: {np.mean(rmse_list)}', flush=True)
    return val_predict

import pickle
with open('../input/train_ridge.p', 'rb') as f:
    train = pickle.load(f)
    
with open('../input/test_ridge.p', 'rb') as f:
    test = pickle.load(f)    
    

    
train = train.iloc[:10000]    
nfolds=10
fname='vgg_base'
epochs= 30



model = get_model()

val_predict = train_bagging(train, train.deal_probability.values, nfolds)
# print(f"model list length: {len(model_list)}")

# fname = 'des_word_svd_200_char_svd_1000_title_200_resnet50_500_lgb_1fold'


print('storing test prediction', flush=True)
for index in tqdm(range(nfold)):
    model_path = f'../weights/{fname}_fold{index}.hdf5'
    model.load_weights(model_path)
    if index == 0: 
        y_pred = model.predict(x_test)
    else:
        y_pred *= model.predict(x_test)
#         y_pred += model.predict(x_test)
    
y_pred = np.clip(y_pred, 0, 1)
y_pred = y_pred **( 1.0/ (nfold))


print('storing test prediction', flush=True)
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv(f'../output/{fname}_test.csv', index=False)


print('storing oof prediction', flush=True)
train_data = pd.read_csv('../input/train.csv.zip')
label = ['deal_probability']
train_user_ids = train_data.user_id.values
train_item_ids = train_data.item_id.values

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_user_ids.reshape(len(train_user_ids), 1)

val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv(f'../output/{fname}_train.csv', index=False)
