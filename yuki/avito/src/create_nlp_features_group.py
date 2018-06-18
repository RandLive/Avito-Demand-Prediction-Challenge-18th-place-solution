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
from scipy.sparse import hstack, csr_matrix, lil_matrix
from nltk.corpus import stopwords
russian_stop = set(stopwords.words('russian'))
stop_2 = set([w for w in open("../tmp/russian_stopwords.txt", "r").readlines()])
russian_stop = russian_stop.union(stop_2)
import string
punctuations = string.punctuation
from nltk.stem.snowball import SnowballStemmer

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
training = pd.read_csv('../input/train.csv', parse_dates = ["activation_date"])
testing = pd.read_csv('../input/test.csv', parse_dates = ["activation_date"])
train_active = pd.read_csv('../input/train_active.csv', parse_dates = ["activation_date"])
test_active = pd.read_csv('../input/test_active.csv', parse_dates = ["activation_date"])

ntrain = training.shape[0]
ntest = testing.shape[0]
y = training.deal_probability.copy()

print("Combine Train and Test")
df = pd.concat([training,testing,train_active,test_active],axis=0)
df.drop_duplicates(['item_id'], inplace=True)
df["text_all"] = df.description.fillna(" ") + " "  + df.title.fillna(" ")
itemid2idx = {item:i for i, item in enumerate(pd.concat([training,testing]).item_id)}
del training, testing, train_active, test_active;gc.collect()
# Stemming
def stemming_text(text):
    stemmer = SnowballStemmer("russian")
    out = []
    for t in text.split():
        stemed = stemmer.stem(t)
        if stemed not in russian_stop and stemed not in punctuations:
            out.append(stemed)
    return " ".join(out)

df["text_all"] = Parallel(n_jobs=6,verbose=1)([delayed(stemming_text)(x) for x in df.text_all.fillna("")])

gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

def get_svd(texts,num):
    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        "min_df":1,
        "max_df":.9,
        "smooth_idf":False
    }
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, **tfidf_para)

    vecs = vectorizer.fit_transform(texts)
    # tfidf dimensionality reduction
    ## NMF
    nmf = NMF(n_components=5)
    X_nmf = nmf.fit_transform(vecs)
    return X_nmf[:num,:]

def get_lda(texts,num):
    text = [[w for w in t.split()] for t in texts]
    len_data = len(texts)
    num_topics = 5
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(t) for t in text]
    print("train lda model")
    lda = models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary)
    vecs = np.zeros((len(corpus), lda.num_topics))
    for i, topics_per_document in enumerate(lda[corpus][:num]):
        for topic_num, prob in topics_per_document:
            vecs[i, topic_num] = prob
    return vecs

dims = 5
df["parent_category_name"] = df["parent_category_name"].fillna("NAN")
df["category_name"] = df["category_name"].fillna("NAN")
parent_categories = df.parent_category_name.unique()
categories = df.category_name.unique()
X = lil_matrix((len(itemid2idx), 5*(len(parent_categories)+len(categories))))
cnt = 0
for cate in parent_categories:
    print(cate)
    df_tmp = df[df.parent_category_name==cate]
    idx = [itemid2idx[i] for i in df_tmp.item_id if i in itemid2idx]
    try:
        X[idx, cnt*5:cnt*5+5] = get_svd(df_tmp.text_all.values,len(idx))
    except:
        X[idx, cnt*5:cnt*5+5] = np.zeros(5)
    print("done NMF")
    # X[idx, cnt*10+5:cnt*10+10] = get_lda(df_tmp.text_all.values,len(idx))
    # print("done lda")
    cnt += 1

for cate in categories:
    print(cate)
    df_tmp = df[df.category_name==cate]
    idx = [itemid2idx[i] for i in df_tmp.item_id if i in itemid2idx]
    try:
        X[idx, cnt*5:cnt*5+5] = get_svd(df_tmp.text_all.values,len(idx))
    except:
        X[idx, cnt*5:cnt*5+5] = np.zeros(5)
    # X[idx, cnt*10+5:cnt*10+10] = get_lda(df_tmp.text_all.values,len(idx))
    cnt += 1

X = X.tocsr()
oof_lgbm(X[:ntrain], X[ntrain:], y, "category_group_nmf")
oof_sgd(X[:ntrain], X[ntrain:], y, "category_group_nmf")
