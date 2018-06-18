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
import seaborn as sns
import matplotlib.pyplot as plt
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
    if col!="text_all":
        continue
    print("Creating features {}...".format(col))
    df_target = df[[col]].fillna("")

    # basic features
    print("Creating basic NLP features...")
    df_out = pd.DataFrame()
    df_out[col + "_count_upper"] = Parallel(n_jobs=-1)([delayed(count_upper)(x) for x in df_target.fillna("")])
    df_out[col + "_count_lower"] = Parallel(n_jobs=-1)([delayed(count_lower)(x) for x in df_target.fillna("")])
    df_out[col + "_count_lower"] += 1
    df_out[col + "_count_upper_lower_ratio"] = df_out[col + "_count_upper"] / df_out[col + "_count_lower"]
    df_target[col] = df_target[col].str.lower()
    df_out[col + '_num_stopwords'] = Parallel(n_jobs=-1)([delayed(count_stops)(x) for x in df_target.fillna("")])
    df_out[col + '_num_punctuations'] = Parallel(n_jobs=-1)([delayed(count_punctuations)(x) for x in df_target.fillna("")])
    df_out[col + '_num_number'] = Parallel(n_jobs=-1)([delayed(count_numbers)(x) for x in df_target.fillna("")])
    df_out[col + '_num_chars'] = Parallel(n_jobs=-1)([delayed(len)(x) for x in df_target.fillna("")])
    df_out[col + '_num_words'] = Parallel(n_jobs=-1)([delayed(count_words)(x) for x in df_target.fillna("")])
    df_out[col + '_num_words'] += 1
    df_out[col + '_num_unique_words'] = Parallel(n_jobs=-1)([delayed(count_unique_words)(x) for x in df_target.fillna("")])
    df_out[col + '_words_vs_unique'] = df_out[col+'_num_unique_words'] / df_out[col+'_num_words']
    to_parquet(df_out.iloc[:ntrain, :], "../features/fe_basic_nlp_{}_train.parquet".format(col))
    to_parquet(df_out.iloc[ntrain:, :], "../features/fe_basic_nlp_{}_test.parquet".format(col))

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
        "max_df":.9,
        "smooth_idf":False
    }
    if "param" not in col:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, **tfidf_para)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=50000)
    vecs = vectorizer.fit_transform(df_target.values)


    df_out = pd.DataFrame()
    df_out[col + "_tfidf_mean"] = Parallel(n_jobs=-1)([delayed(np.mean)(v) for v in vecs])
    print("mean done")
    df_out[col + "_tfidf_max"] = Parallel(n_jobs=-1)([delayed(np.max)(v) for v in vecs])
    print("max done")
    df_out[col + "_tfidf_min"] = Parallel(n_jobs=-1)([delayed(np.min)(v) for v in vecs])
    print("min done")
    df_out[col + "_tfidf_std"] = Parallel(n_jobs=-1)([delayed(get_std)(v) for v in vecs])
    print("std done")
    df_out[col + "_tfidf_skew"] = Parallel(n_jobs=-1)([delayed(get_skew)(v) for v in vecs])
    print("skew done")
    df_out[col + "_tfidf_kur"] = Parallel(n_jobs=-1)([delayed(get_kur)(v) for v in vecs])
    print("kurtoisis done")
    df_out[col + "_tfidf_entropy"] = Parallel(n_jobs=-1)([delayed(get_entropy)(v) for v in vecs])
    print("entropy done")
    df_out[col + "_tfidf_sum"] = Parallel(n_jobs=-1)([delayed(np.sum)(v) for v in vecs])
    print("sum done")
    to_parquet(df_out.iloc[:ntrain, :], "../features/fe_tfidf_basic_{}_train.parquet".format(col))
    to_parquet(df_out.iloc[ntrain:, :], "../features/fe_tfidf_basic_{}_test.parquet".format(col))
    del df_out; gc.collect()

    oof_sgd(vecs[:ntrain,:],vecs[ntrain:,:],y,"tfidf_{}".format(col))
    oof_lgbm(vecs[:ntrain,:].astype(np.float32),vecs[ntrain:,:].astype(np.float32),y,"tfidf_{}".format(col))
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
        "max_df":.9,
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
    train_svd = pd.DataFrame(m_svd[:ntrain, :], columns=["svd_{}_1".format(col+"_tfidf"), "svd_{}_2".format(col+"_tfidf"), "svd_{}_3".format(col+"_tfidf")])
    test_svd = pd.DataFrame(m_svd[ntrain:, :], columns=["svd_{}_1".format(col+"_tfidf"), "svd_{}_2".format(col+"_tfidf"), "svd_{}_3".format(col+"_tfidf")])
    to_parquet(train_svd, "../features/fe_tfidf_svd_{}_train.parquet".format(col))
    to_parquet(test_svd, "../features/fe_tfidf_svd_{}_test.parquet".format(col))
    del m_svd, train_svd, test_svd; gc.collect()

    ## NMF
    nmf = NMF(n_components=3)
    X_nmf = nmf.fit_transform(vecs)
    df_nmf = pd.DataFrame(X_nmf, columns=["nmf_{}_1".format(col+"_tfidf"), "nmf_{}_2".format(col+"_tfidf"), "nmf_{}_3".format(col+"_tfidf")])
    nmf_train = df_nmf.iloc[:ntrain,:]
    nmf_test = df_nmf.iloc[ntrain:,:]
    to_parquet(nmf_train, "../features/fe_tfidf_nmf_{}_train.parquet".format(col))
    to_parquet(nmf_test, "../features/fe_tfidf_nmf_{}_test.parquet".format(col))
    del df_nmf, nmf_train, nmf_test; gc.collect()

    # LDA
    text = [[w for w in t.split() if w not in russian_stop and w not in punctuations] for t in df_target]
    len_data = df_target.shape[0]
    num_topics = 8
    dictionary = corpora.Dictionary(text)
    dictionary.save("../model/gensim/dictionary.dict")
    corpus = [dictionary.doc2bow(t) for t in text]
    corpora.MmCorpus.serialize("../model/gensim/tokens.mm", corpus)
    print("train lda model")
    if not os.path.exists("../model/gensim/lda_{}.model".format(col)):
        lda = models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=3000)
        lda.save("../model/gensim/lda_{}.model".format(col))
    else:
        lda = models.ldamulticore.LdaMulticore.load("../model/gensim/lda_{}.model".format(col))
    vecs = np.zeros((len(corpus), lda.num_topics))
    for i, topics_per_document in enumerate(lda[corpus]):
        for topic_num, prob in topics_per_document:
            vecs[i, topic_num] = prob

    df_out = pd.DataFrame(vecs,columns=["lda_{}_{}".format(col, i+1) for i in range(num_topics)])
    to_parquet(df_out.iloc[:ntrain, :], "../features/fe_lda{}_{}_train.parquet".format(num_topics,col))
    to_parquet(df_out.iloc[ntrain:, :], "../features/fe_lda{}_{}_test.parquet".format(num_topics,col))

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
        "max_df":.9,
        "smooth_idf":False
    }
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, **tfidf_para)

    vecs = vectorizer.fit_transform(df_target.values)
    ## UMAP
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation', n_components=3).fit_transform(vecs)
    df_umap = pd.DataFrame(embedding, columns=["umap_{}_1".format(col+"_tfidf"), "umap_{}_2".format(col+"_tfidf"), "umap_{}_3".format(col+"_tfidf")])
    umap_train = df_umap.iloc[:ntrain,:]
    umap_test = df_umap.iloc[ntrain:,:]
    to_parquet(umap_train, "../features/fe_tfidf_umap_{}_train.parquet".format(col))
    to_parquet(umap_test, "../features/fe_tfidf_umap_{}_test.parquet".format(col))
    del df_umap, umap_train, umap_test; gc.collect()
