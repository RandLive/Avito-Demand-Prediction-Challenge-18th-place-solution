import numpy as np
import keras
import cv2
import os.path
import os
import subprocess 
from joblib import Parallel, delayed

class ImageGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dir_, item_ids, image_ids, labels, batch_size=32, dim=(160,160), n_channels=3, shuffle=True):
        'Initialization'
        self.dir = dir_
        self.dim = dim
        self.batch_size = batch_size
        # labels index by item id
        self.labels = labels
        # image id index by item id
        self.image_ids = image_ids
        self.item_ids = item_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.item_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        item_ids_temp = [self.item_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(item_ids_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.item_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _load_image(self, item_id):
        image_id = self.image_ids[item_id]
        try:
            fname = f'{self.dir}/{image_id}.jpg'
            img = cv2.imread(fname)
            img = cv2.resize(img, self.dim, interpolation = cv2.INTER_LINEAR)
            return img
        except cv2.error as e:
            return np.zeros([*self.dim, self.n_channels])
        except:
            return np.zeros([*self.dim, self.n_channels])
        
    def _get_label(self,item_id):
        return self.labels[item_id]
    def __data_generation(self, item_ids_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=np.float32)

        # Generate data
        for i, item_id in enumerate(item_ids_temp):
            image_id = self.image_ids[item_id]
            # Store sample
           
            fname = f'{self.dir}/{image_id}.jpg'
            if os.path.isfile(fname):
                img = cv2.imread(fname)
            else: 
                img = np.zeros([*self.dim, self.n_channels])
            img = cv2.resize(img, self.dim, interpolation = cv2.INTER_LINEAR)
            X[i,] = img
            y[i] = self.labels[item_id]
#         parallel = Parallel(self.batch_size, backend="threading", verbose=0)
#         X = parallel(delayed(self._load_image)(item_id) for item_id in item_ids_temp)
#         y = parallel(delayed(self._get_label)(item_id) for item_id in item_ids_temp)
#         X = np.array(X)

        return X, y
    
    