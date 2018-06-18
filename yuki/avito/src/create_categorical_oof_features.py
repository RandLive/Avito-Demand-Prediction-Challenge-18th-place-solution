# In this script, we will create some basic nlp features.
import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import NMF

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
russian_stop = set(stopwords.words('russian'))
stop_2 = set([w for w in open("../tmp/russian_stopwords.txt", "r").readlines()])
russian_stop = russian_stop.union(stop_2)
import string
punctuations = string.punctuation

# Viz
import re
import string

# I/O
from utils import *

# others
from joblib import Parallel, delayed


NFOLDS = 5
SEED = 42

print("\nData Load Stage")
training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)

del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

cate_cols = ["parent_category_name","category_name", "user_type","image_top_1", "param_1", "param_2", "param_3", "city", "region"]
for col in cate_cols:
    print("Creating features {}...".format(col))
    df_target = df[col].astype(str).fillna("NAN").values
    # basic features
    lbl = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder(sparse=True)
    vecs = ohe.fit_transform(lbl.fit_transform(df_target).reshape(df_target.shape[0], 1))
    if col=="parent_category_name":
        all_vecs = vecs
    else:
        all_vecs = hstack([all_vecs, vecs])


    # oof_sgd(vecs[:ntrain,:],vecs[ntrain:,:],y,"categorical_ohe_{}".format(col))
    # oof_lgbm(vecs[:ntrain,:],vecs[ntrain:,:],y,"categorical_ohe_{}".format(col))
all_vecs = all_vecs.tocsr()
oof_sgd(all_vecs[:ntrain,:],all_vecs[ntrain:,:],y,"categorical_ohe_{}".format("all_categories"))
oof_lgbm(all_vecs[:ntrain,:],all_vecs[ntrain:,:],y,"categorical_ohe_{}".format("all_categories"))
