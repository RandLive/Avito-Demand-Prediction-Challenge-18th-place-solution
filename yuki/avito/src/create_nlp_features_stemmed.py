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
from scipy.stats import skew, kurtosis, entropy
from scipy import sparse
from scipy.sparse.linalg import svds
import umap
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim import corpora, models

NFOLDS = 5
SEED = 42

print("\nData Load Stage")
training = pd.read_csv('../input/train_stemmed.csv', index_col = "item_id")
traindex = training.index
testing = pd.read_csv('../input/test_stemmed.csv', index_col = "item_id")
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

y = pd.read_csv('../input/train.csv', usecols = ["deal_probability"]).deal_probability.values

print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
df["text_all"] = df.description.fillna("") + " "  + df.title.fillna("")
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

# Define Functions
def count_upper(x): return len([w for w in x if w.isupper()])
def count_lower(x): return len([w for w in x if w.islower()])
def count_stops(x): return len([w for w in x if w in russian_stop])
def count_punctuations(x): return len([w for w in x if w in punctuations])
def count_numbers(x): return len([w for w in x if w.isnumeric()])
def count_words(x): return len(x.split())
def count_unique_words(x): return len(set(x.split()))
def get_std(x): return np.std(x.todense())
def get_skew(x): return skew(x.todense().T)[0]
def get_kur(x): return kurtosis(x.todense().T)[0]
def get_entropy(x): return entropy(x.todense().T)[0]


textcols = ["description", "title", "text_all"]#"description"
for col in textcols:
    print("Creating features {}...".format(col))
    df_target = df[[col]].fillna("")

    print("Creating basic tfidf features...")
    df_target = df[col].fillna("")
    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        "min_df":3,
        "smooth_idf":True
    }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=17000, **tfidf_para)
    vecs = vectorizer.fit_transform(df_target.values)

    df_out = pd.DataFrame()
    df_out[col + "_stemmed_tfidf_mean"] = Parallel(n_jobs=-1)([delayed(np.mean)(v) for v in vecs])
    print("mean done")
    df_out[col + "_stemmed_tfidf_max"] = Parallel(n_jobs=-1)([delayed(np.max)(v) for v in vecs])
    print("max done")
    df_out[col + "_stemmed_tfidf_min"] = Parallel(n_jobs=-1)([delayed(np.min)(v) for v in vecs])
    print("min done")
    df_out[col + "_stemmed_tfidf_std"] = Parallel(n_jobs=-1)([delayed(get_std)(v) for v in vecs])
    print("std done")
    df_out[col + "_stemmed_tfidf_skew"] = Parallel(n_jobs=-1)([delayed(get_skew)(v) for v in vecs])
    print("skew done")
    df_out[col + "_stemmed_tfidf_kur"] = Parallel(n_jobs=-1)([delayed(get_kur)(v) for v in vecs])
    print("kurtoisis done")
    df_out[col + "_stemmed_tfidf_entropy"] = Parallel(n_jobs=-1)([delayed(get_entropy)(v) for v in vecs])
    print("entropy done")
    df_out[col + "_stemmed_tfidf_sum"] = Parallel(n_jobs=-1)([delayed(np.sum)(v) for v in vecs])
    print("sum done")
    to_parquet(df_out.iloc[:ntrain, :], "../features/fe_tfidf_stemmed_basic_{}_train.parquet".format(col))
    to_parquet(df_out.iloc[ntrain:, :], "../features/fe_tfidf_stemmed_basic_{}_test.parquet".format(col))
    del df_out; gc.collect()

    oof_sgd(vecs[:ntrain,:],vecs[ntrain:,:],y,"tfidf_stemmed_{}".format(col))
    oof_lgbm(vecs[:ntrain,:].astype(np.float32),vecs[ntrain:,:].astype(np.float32),y,"tfidf_stemmed_{}".format(col))
    del vecs; gc.collect()

for col in textcols:
    df_target = df[col].fillna("")
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
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, **tfidf_para)

    vecs = vectorizer.fit_transform(df_target.values)
    # tfidf dimensionality reduction
    print("Start dimensionality reduction")
    ## SVD
    U, S , _ = svds(vecs.tocsc(),k=3)
    m_svd = [U[i] * S for i in range(U.shape[0])]
    m_svd = np.array(m_svd)
    train_svd = pd.DataFrame(m_svd[:ntrain, :], columns=["svd_stemmed_{}_1".format(col+"_tfidf"), "svd_stemmed_{}_2".format(col+"_tfidf"), "svd_stemmed_{}_3".format(col+"_tfidf")])
    test_svd = pd.DataFrame(m_svd[ntrain:, :], columns=["svd_stemmed_{}_1".format(col+"_tfidf"), "svd_stemmed_{}_2".format(col+"_tfidf"), "svd_stemmed_{}_3".format(col+"_tfidf")])
    to_parquet(train_svd, "../features/fe_tfidf_stemmed_svd_{}_train.parquet".format(col))
    to_parquet(test_svd, "../features/fe_tfidf_stemmed_svd_{}_test.parquet".format(col))
    del m_svd, train_svd, test_svd; gc.collect()

    ## NMF
    nmf = NMF(n_components=3)
    X_nmf = nmf.fit_transform(vecs)
    df_nmf = pd.DataFrame(X_nmf, columns=["nmf_stemmed_{}_1".format(col+"_tfidf"), "nmf_stemmed_{}_2".format(col+"_tfidf"), "nmf_stemmed_{}_3".format(col+"_tfidf")])
    nmf_train = df_nmf.iloc[:ntrain,:]
    nmf_test = df_nmf.iloc[ntrain:,:]
    to_parquet(nmf_train, "../features/fe_tfidf_stemmed_nmf_{}_train.parquet".format(col))
    to_parquet(nmf_test, "../features/fe_tfidf_stemmed_nmf_{}_test.parquet".format(col))
    del df_nmf, nmf_train, nmf_test; gc.collect()


# Meta Text Features

with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)

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

df['title'] = df['title'].apply(lambda x: cleanName(x)).fillna("")
df["description"]   = df["description"].apply(lambda x: cleanName(x)).fillna("")

# textfeats = ["description", "title"]
# tfidf_para = {
#     "stop_words": russian_stop,
#     "analyzer": 'word',
#     "token_pattern": r'\w{1,}',
#     "sublinear_tf": True,
#     "dtype": np.float32,
#     "norm": 'l2',
#     "min_df":3,
#     "smooth_idf":False
# }
#
# def get_col(col_name): return lambda x: x[col_name]
#
# vectorizer = FeatureUnion([
#         ('description',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=17000,
#             **tfidf_para,
#             preprocessor=get_col('description'))),
#         ('title',CountVectorizer(
#             ngram_range=(1, 2),
#             stop_words = russian_stop,
#             preprocessor=get_col('title')))
#     ])
#
# #Fit my vectorizer on the entire dataset instead of the training rows
# #Score improved by .0001
# vectorizer.fit(df.to_dict('records'))
#
# ready_df = vectorizer.transform(df.to_dict('records'))
# tfvocab = vectorizer.get_feature_names()
#
# # Drop Text Cols
# textfeats = ["description", "title"]
# df.drop(textfeats, axis=1,inplace=True)
#
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
#                 'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
#
# ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
# ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
# train_out = pd.DataFrame(ridge_oof_train,columns=["ridge_oof_base_stemmed_2"])
# test_out = pd.DataFrame(ridge_oof_test,columns=["ridge_oof_base_stemmed_2"])
#
# to_parquet(train_out, "../features/oof_ridge_tfidf_stemmed_2_train.parquet")
# to_parquet(test_out, "../features/oof_ridge_tfidf_stemmed_2_test.parquet")
#


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
            max_features=17000,
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

ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])
train_out = pd.DataFrame(ridge_oof_train,columns=["ridge_oof_base_stemmed_3"])
test_out = pd.DataFrame(ridge_oof_test,columns=["ridge_oof_base_stemmed_3"])

to_parquet(train_out, "../features/oof_ridge_tfidf_stemmed_3_train.parquet")
to_parquet(test_out, "../features/oof_ridge_tfidf_stemmed_3_test.parquet")
