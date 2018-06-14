import glob
import cv2
import os

from joblib import Parallel, delayed


train_dir = '../input/train_jpg/data/competition_files/train_jpg/'
test_dir = '../input/test_jpg/data/competition_files/test_jpg/'

train_out_dir = '../input/train_jpg/data/competition_files/train_jpg_ds/'
test_out_dir = '../input/test_jpg/data/competition_files/test_jpg_ds/'

os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(test_out_dir, exist_ok=True)

dim = (160,160)


print('getting id')
train_ids = os.listdir(train_dir)
test_ids = os.listdir(test_dir)

print('ids got')
def resize_store(image_id, in_dir, out_dir ):
    try:
        in_name = f'{in_dir}/{image_id}'
        img = cv2.imread(in_name)
        img = cv2.resize(img, dim,  interpolation = cv2.INTER_LINEAR)
        out_name = f'{out_dir}/{image_id}'
        cv2.imwrite(out_name, img)
    except cv2.error as e:
        print(e)
    except:
        print('Unknown error')
        
parallel = Parallel(2, backend="threading", verbose=100)        
parallel(delayed(resize_store)(id_, train_dir, train_out_dir) for id_ in train_ids)
parallel(delayed(resize_store)(id_, test_dir, test_out_dir) for id_ in test_ids)
        