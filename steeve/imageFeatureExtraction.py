
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import pickle

# In[2]:



# In[3]:


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# restrict gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# In[4]:




# In[45]:


from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
# import numpy as np

print('loading model... ')
model = Xception(include_top=True, weights='imagenet')
# model.load_weights('../weights/xeption_weights_tf_dim_ordering_tf_kernels.h5')


# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.inception_v3 import preprocess_input


# print('loading model... ')
# model = InceptionV3(weights=None)
# model.load_weights('../weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

# In[7]:


from keras.preprocessing import image

from tqdm import tqdm
image_dir = '../input/test_jpg/data/competition_files/test_jpg/'

print('getting img names')
# In[5]:


img_names = [i for i in os.walk(image_dir)][0][2]
print(f'size of images: {len(img_names)}')

features = []
ids = []
for img_name in tqdm(img_names):
    
    img_path = image_dir + img_name
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except OSError:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # 4 dims(1, 3, 299, 299)
    x = preprocess_input(x)
    
    feature = model.predict(x)
#     print(feature.shape)
    id_ = img_name[:-4]
    features.append(feature)
    ids.append(id_)
    # print(feature.shape)


x = {}
x['features'] = features
x['ids'] = ids
with open('../input/xeption_features_test.p','wb') as f:
    pickle.dump(x, f)  
    

del x
del features
del ids
del img_names
import gc
gc.collect()

image_dir = '../input/train_jpg/data/competition_files/train_jpg/'

print('getting img names')
# In[5]:


img_names = [i for i in os.walk(image_dir)][0][2]
print(f'size of images: {len(img_names)}')

features = []
ids = []
for img_name in tqdm(img_names):
    
    img_path = image_dir + img_name
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except OSError:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # 4 dims(1, 3, 299, 299)
    x = preprocess_input(x)
    
    feature = model.predict(x)
#     print(feature.shape)
    id_ = img_name[:-4]
    features.append(feature)
    ids.append(id_)
    # print(feature.shape)


x = {}
x['features'] = features
x['ids'] = ids
with open('../input/xeption_features_train.p','wb') as f:
    pickle.dump(x, f)  
    

del x
del features
del ids
del img_names
import gc
gc.collect()