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

print("Feature Engineering")
# NAN features
df_out = pd.DataFrame()
nan_cols = ["description", "image", "param_1", "param_2", "param_3", "price"]
for cols in nan_cols:
    df_out[cols + "_is_NAN_bool"] = df[cols].fillna("MISSINGGGGGGGGGGGGGGGGG").apply(lambda x: int(x=="MISSINGGGGGGGGGGGGGGGGG"))
df_out["num_NAN"] = df_out.sum(axis=1)
to_parquet(df_out.iloc[:ntrain,:], "../features/fe_nan_features_train.parquet")
to_parquet(df_out.iloc[ntrain:,:], "../features/fe_nan_features_test.parquet")

# Base Features
print("Creating Base Features...")
df_out = pd.DataFrame()
df_out["price_filled"] = df["price"].fillna(df["price"].median())
df_out["price"] = df["price"]
df_out["log_price"] = np.log(df["price"]+0.0001)
df["image_top_1"] = df["image_top_1"].fillna(-1)
df_out["item_seq_number"] = df["item_seq_number"]

print("\nCreate Time Variables")
df_out["Weekday"] = df['activation_date'].dt.weekday
# additional
tmp = df_out.groupby("Weekday").price.agg(np.median).reset_index().rename(columns={"price":"weekday_price_median"})
df_out = pd.merge(df_out, tmp, on="Weekday", how="left")
df_out.drop("price", axis=1, inplace=True)
del tmp; gc.collect()

categorical = ["region","city","parent_category_name","category_name","user_type","image_top_1"]
print("Encoding :",categorical)

# Encoder:
# lbl = preprocessing.LabelEncoder()
# for col in categorical:
#     df_out[col+"_labelencoding"] = lbl.fit_transform(df[col].astype(str))

print(df_out.dtypes)
to_parquet(df_out.iloc[:ntrain,:], "../features/fe_base_features_train.parquet")
to_parquet(df_out.iloc[ntrain:,:], "../features/fe_base_features_test.parquet")

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

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']),
    str(row['param_2']),
    str(row['param_3'])]),axis=1).fillna("missing") # Group Param Features

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
textfeats = ["description","text_feat", "title"]
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "smooth_idf":False
}

def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            preprocessor=get_col('title')))
    ])

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

# Drop Text Cols
textfeats = ["description","text_feat", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
train_out = pd.DataFrame(ridge_oof_train,columns=["ridge_oof_base"])
test_out = pd.DataFrame(ridge_oof_test,columns=["ridge_oof_base"])
print(train_out.columns)
print(test_out.columns)
to_parquet(train_out, "../features/oof_ridge_tfidf_train.parquet")
to_parquet(test_out, "../features/oof_ridge_tfidf_test.parquet")
