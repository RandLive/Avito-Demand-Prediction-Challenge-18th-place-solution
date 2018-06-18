# fasttext sentence vector oof, dimentionality reduction, w2v features
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
import re
import string

# I/O
from utils import *
import umap
from joblib import Parallel, delayed
from scipy.stats import skew, kurtosis, entropy
import fastText
import gensim

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
df["text_all"] = df.description.fillna("") + " " + df.title.fillna("")
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
# util functions

print("Fasttext embedding")

# vec_train = read_parquet("../vectors/sentence_vectors_without_stop_train.parquet").values
# vec_test = read_parquet("../vectors/sentence_vectors_without_stop_test.parquet").values[:ntest,:]
# oof_sgd(vec_train,vec_test,y,"oof_sentencevector")
# oof_lgbm(vec_train,vec_test,y,"oof_sentencevector")
# vecs = np.concatenate([vec_train, vec_test])
# df_out = pd.DataFrame()
# df_out["sentencevector_mean"] = np.mean(vecs,axis=1)
# print("mean done")
# df_out["sentencevector_max"] = np.max(vecs,axis=1)
# print("max done")
# df_out["sentencevector_min"] = np.min(vecs,axis=1)
# print("min done")
# df_out["sentencevector_std"] = np.std(vecs, axis=1)
# print("std done")
# df_out["sentencevector_skew"] = skew(vecs, axis=1)
# print("skew done")
# df_out["sentencevector_kur"] = kurtosis(vecs, axis=1)
# print("kurtoisis done")
# df_out["sentencevector_entropy"] = Parallel(n_jobs=-1)([delayed(entropy)(x) for x in vecs])
# print("entropy done")
# df_out["sentencevector_sum"] = np.sum(vecs, axis=1)
# to_parquet(df_out.iloc[:ntrain, :], "../features/fe_sentencevector_basic_train.parquet")
# to_parquet(df_out.iloc[ntrain:, :], "../features/fe_sentencevector_basic_test.parquet")
# del df_out; gc.collect()

# w2v model
model = fastText.load_model("../model/fasttext_model_without_stop.bin")
def sent2vec(words, dim):
    M = []
    for w in words:
        M.append(model.get_word_vector(w))
    if len(M)==0:
        return np.zeros(dim)
    else:
        M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

def sent2vec_min(words, dim):
    M = []
    for w in words:
        M.append(model.get_word_vector(w))
    M = np.array(M)
    if len(M) != 0:
        v = M.min(axis=0)
    else:
        return np.zeros(dim)
    return v

def sent2vec_max(words, dim):
    M = []
    for w in words:
        M.append(model.get_word_vector(w))
    M = np.array(M)
    if len(M)!=0:
        v = M.max(axis=0)
    else:
        return np.zeros(dim)
    return v

text = df["text_all"].apply(lambda x: x.split()).values
train = text[:ntrain]
test = text[ntrain:]
del text; gc.collect()
print("text...")

train_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec)(s, 100) for s in train]))
print("train load...")
test_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec)(s, 100) for s in test]))
print("test load...")
oof_sgd(train_vectors,test_vectors,y,"meanvectors_fasttext")
oof_lgbm(train_vectors,test_vectors,y,"meanvectors_fasttext")

train_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_min)(s, 100) for s in train]))
test_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_min)(s, 100) for s in test]))
oof_sgd(train_vectors,test_vectors,y,"minvectors_fasttext")
oof_lgbm(train_vectors,test_vectors,y,"minvectors_fasttext")

train_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_max)(s, 100) for s in train]))
test_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_max)(s, 100) for s in test]))
oof_sgd(train_vectors,test_vectors,y,"maxvectors_fasttext")
oof_lgbm(train_vectors,test_vectors,y,"maxvectors_fasttext")

# vec_train = read_parquet("../vectors/sentence_vectors_without_stop_train.parquet").values
# vec_test = read_parquet("../vectors/sentence_vectors_without_stop_test.parquet").values[:ntest,:]
# print("Read vectors...")
# vecs = np.concatenate([vec_train, vec_test])
# print("Concatenated vectors...")
# del vec_train, vec_test; gc.collect()
# print("Start training")
# embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='correlation', n_components=3).fit_transform(vecs)
# print("Training Done")
# df_umap = pd.DataFrame(embedding, columns=["umap_{}_1".format("sentencevector"), "umap_{}_2".format("sentencevector"), "umap_{}_3".format("sentencevector")])
# print("Turning into DataFrame")
# umap_train = df_umap.iloc[:ntrain,:]
# umap_test = df_umap.iloc[ntrain:,:]
# print("Saving...")
# to_parquet(umap_train, "../features/fe_sentencevector_umap_train.parquet")
# to_parquet(umap_test, "../features/fe_sentencevector_umap_test.parquet")
# print("Save Done")
# del df_umap, umap_train, umap_test; gc.collect()

#
# print("\nData Load Stage")
# training = pd.read_csv('../input/train_stemmed.csv')
# traindex = training.index
# testing = pd.read_csv('../input/test_stemmed.csv')
# testdex = testing.index
#
# ntrain = training.shape[0]
# ntest = testing.shape[0]
#
# y = pd.read_csv('../input/train.csv',usecols=["deal_probability"]).deal_probability.copy()
# print('Train shape: {} Rows, {} Columns'.format(*training.shape))
# print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
#
# print("Combine Train and Test")
# df = pd.concat([training,testing],axis=0)
# df["text_all"] = df.description.fillna("") + " " + df.title.fillna("")
# del training, testing
# gc.collect()
# print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
# # util functions
#
# model = gensim.models.KeyedVectors.load_word2vec_format("../model/wiki.ru.vec",binary=False)
#
# def word2vec_sentencevec(words, dim):
#     M = []
#     for w in words.split():
#         try:
#             M.append(model[w])
#         except:
#             M.append(np.zeros(dim))
#     if len(M)==0:
#         return np.zeros(dim)
#     else:
#         M = np.array(M)
#     v = M.sum(axis=0)
#     v = np.array(v)
#     return v / np.sqrt((v ** 2).sum())
#
# text = df["text_all"].values
# train = text[:ntrain]
# test = text[ntrain:]
# del text; gc.collect()
# print("text...")
#
# train_vectors = np.array(Parallel(n_jobs=2,verbose=2)([delayed(word2vec_sentencevec)(s, 300) for s in train]))
# print("train load...")
# del train; gc.collect()
# test_vectors = np.array(Parallel(n_jobs=2,verbose=2)([delayed(word2vec_sentencevec)(s, 300) for s in test]))
# print("test load...")
# del test; gc.collect()
# # oof_sgd(train_vectors,test_vectors,y,"stemmed_wordvector")
# oof_lgbm(train_vectors,test_vectors,y,"stemmed_wordvector")
#
# vecs = np.concatenate([train_vectors, test_vectors])
# del train_vectors, test_vectors;gc.collect()
# df_out = pd.DataFrame()
# df_out["stemmed_wordvector_mean"] = np.mean(vecs,axis=1)
# print("mean done")
# df_out["stemmed_wordvector_max"] = np.max(vecs,axis=1)
# print("max done")
# df_out["stemmed_wordvector_min"] = np.min(vecs,axis=1)
# print("min done")
# df_out["stemmed_wordvector_std"] = np.std(vecs, axis=1)
# print("std done")
# df_out["stemmed_wordvector_skew"] = skew(vecs, axis=1)
# print("skew done")
# df_out["stemmed_wordvector_kur"] = kurtosis(vecs, axis=1)
# print("kurtoisis done")
# df_out["stemmed_wordvector_entropy"] = Parallel(n_jobs=-1)([delayed(entropy)(x) for x in vecs])
# print("entropy done")
# df_out["stemmed_wordvector_sum"] = np.sum(vecs, axis=1)
# to_parquet(df_out.iloc[:ntrain, :], "../features/fe_stemmed_wordvector_basic_train.parquet")
# to_parquet(df_out.iloc[ntrain:, :], "../features/fe_stemmed_wordvector_basic_test.parquet")
