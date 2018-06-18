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
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

# I/O
from utils import *


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

# RIDGE tfidf oof
# https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
if os.path.exists("../tmp/oof_index.dat"):
    with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)
else:
    dftrain_tmp = pd.read_csv("../input/train.csv")
    fold = KFold(n_splits=5, shuffle=True, random_state=1234)
    kfolds = list(fold.split(dftrain_tmp))
    with open("../tmp/oof_index.dat", "wb") as f:
        dill.dump(kfolds, f)
    del dftrain_tmp; gc.collect()

print("Creating Ridge Features...")
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kfolds):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



def cleanName(text):
    try:
        textProc = text.lower()
        textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        regex = re.compile(u'[^[:alpha:]]')
        textProc = regex.sub(" ", textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except:
        return "name error"

df['title'] = df['title'].apply(lambda x: cleanName(x)).fillna("missing")
df["description"]   = df["description"].apply(lambda x: cleanName(x)).fillna("missing")


# Meta Text Features
textfeats = ["description", "title"]
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "min_df":3,
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]

vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=30000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            ngram_range=(1, 3),
            stop_words = russian_stop,
            preprocessor=get_col('title')))
    ])

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':25.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
train_out = pd.DataFrame(ridge_oof_train,columns=["ridge_oof_base_3"])
test_out = pd.DataFrame(ridge_oof_test,columns=["ridge_oof_base_3"])
print(train_out.columns)
print(test_out.columns)
to_parquet(train_out, "../features/oof_ridge_tfidf_3_train.parquet")
to_parquet(test_out, "../features/oof_ridge_tfidf_3_test.parquet")
