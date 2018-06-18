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
y_bin = (y==0).astype(int)
y_high = (y>=0.55).astype(int)
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)

del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))
# Categorical OOF
cate_cols = ["parent_category_name","category_name", "user_type","image_top_1", "param_1", "param_2", "param_3", "city", "region"]
for col in cate_cols:
    print("Creating features {}...".format(col))
    df_target = df[col].astype(str).fillna("NAN").values
    lbl = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder(sparse=True)
    vecs = ohe.fit_transform(lbl.fit_transform(df_target).reshape(df_target.shape[0], 1))
    if col=="parent_category_name":
        all_vecs = vecs
    else:
        all_vecs = hstack([all_vecs, vecs])

all_vecs = all_vecs.tocsr()
oof_lgbm_classify(all_vecs[:ntrain,:],all_vecs[ntrain:,:],y_bin,"categorical_ohe_bin_classify_{}".format("all_categories"))
oof_lgbm_classify(all_vecs[:ntrain,:],all_vecs[ntrain:,:],y_high,"categorical_ohe_high_classify_{}".format("all_categories"))

# NLP OOF
textcols = ["text_all", "title"]# "description"
train = pd.read_csv('../input/train_stemmed.csv')
test = pd.read_csv('../input/test_stemmed.csv')
df = pd.concat([train,test],axis=0)
del train, test;gc.collect()
df["text_all"] = df.description.fillna("") + " "  + df.title.fillna("")
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
    oof_lgbm_classify(vecs[:ntrain,:].astype(np.float32),vecs[ntrain:,:].astype(np.float32),y_bin,"tfidf_stemmed_bin_classify_{}".format(col))
    oof_lgbm_classify(vecs[:ntrain,:].astype(np.float32),vecs[ntrain:,:].astype(np.float32),y_high,"tfidf_stemmed_high_classify_{}".format(col))
    del vecs; gc.collect()


# Fasttextmodel
# input file for fasttext(without stopwords, punctuations)
# f = open("../tmp/text_for_fasttext_stemmed.txt", "w")
# for text in df["text_all"].values:
#     text = text.lower()
#     t = ' '.join([t for t in text.split() if t not in russian_stop and t not in punctuations and '\n' != t and " " != t])
#     t = [s for s in t if "\n" != s and '"' != s]
#     text = "".join(t)
#     text = re.sub(r"^\s+","",text)
#     f.write("".join(text))
#     f.write("\n")
# f.close()
#
# os.system("fasttext skipgram -input ../tmp/text_for_fasttext_stemmed.txt -output ../model/fasttext_model_stemmed -dim 100 -minCount 1")
# create sentence vectors
import fastText
text = df["text_all"].values[:ntrain+ntest]
model = fastText.load_model("../model/fasttext_model_stemmed.bin")
sentence = np.zeros((text.shape[0], 100))
for i, t in enumerate(text):
    s = " ".join(t).replace("\n", " ")
    sentence[i, :] = model.get_sentence_vector(s)

oof_lgbm_classify(sentence[:ntrain,:],sentence[ntrain:,:],y_bin,"oof_bin_classify_sentencevector")
oof_lgbm_classify(sentence[:ntrain,:],sentence[ntrain:,:],y_high,"oof_high_classify_sentencevector")



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
oof_lgbm_classify(train_vectors,test_vectors,y_bin,"meanvectors_bin_classify_fasttext")
oof_lgbm_classify(train_vectors,test_vectors,y_high,"meanvectors_high_classify_fasttext")

train_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_min)(s, 100) for s in train]))
test_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_min)(s, 100) for s in test]))
oof_lgbm_classify(train_vectors,test_vectors,y_bin,"minvectors_bin_classify_fasttext")
oof_lgbm_classify(train_vectors,test_vectors,y_high,"minvectors_high_classify_fasttext")

train_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_max)(s, 100) for s in train]))
test_vectors = np.array(Parallel(n_jobs=-1)([delayed(sent2vec_max)(s, 100) for s in test]))
oof_lgbm_classify(train_vectors,test_vectors,y_bin,"maxvectors_bin_classify_fasttext")
oof_lgbm_classify(train_vectors,test_vectors,y_high,"maxvectors_high_classify_fasttext")

model = gensim.models.KeyedVectors.load_word2vec_format("../model/wiki.ru.vec",binary=False)

def word2vec_sentencevec(words, dim):
    M = []
    for w in words.split():
        try:
            M.append(model[w])
        except:
            M.append(np.zeros(dim))
    if len(M)==0:
        return np.zeros(dim)
    else:
        M = np.array(M)
    v = M.sum(axis=0)
    v = np.array(v)
    return v / np.sqrt((v ** 2).sum())

text = df["text_all"].values
train = text[:ntrain]
test = text[ntrain:]
del text; gc.collect()
print("text...")

train_vectors = np.array(Parallel(n_jobs=2,verbose=2)([delayed(word2vec_sentencevec)(s, 300) for s in train]))
print("train load...")
del train; gc.collect()
test_vectors = np.array(Parallel(n_jobs=2,verbose=2)([delayed(word2vec_sentencevec)(s, 300) for s in test]))
print("test load...")
del test; gc.collect()
oof_lgbm_classify(train_vectors,test_vectors,y_bin,"stemmed_wordvector_bin")
oof_lgbm_classify(train_vectors,test_vectors,y_high,"stemmed_wordvector_high")
