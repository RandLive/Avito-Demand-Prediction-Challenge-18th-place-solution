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


NFOLDS = 5
SEED = 42

print("\nData Load Stage")
training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

train_active = pd.read_csv("../input/train_active.csv", index_col = "item_id", parse_dates = ["activation_date"])
test_active = pd.read_csv("../input/test_active.csv", index_col = "item_id", parse_dates = ["activation_date"])

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing,train_active,test_active],axis=0)
del training, testing,train_active,test_active
gc.collect()
df["text_all"] = df.description.fillna("") + " " + df.param_1.fillna("") \
            + " " + df.param_2.fillna("") + " " + df.param_3.fillna("") \
            + " " + df.title.fillna("")
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Fasttext embedding")

# input file for fasttext(without stopwords, punctuations)
f = open("../tmp/text_for_fasttext_without_stop.txt", "w")
for text in df["text_all"].values:
    text = text.lower()
    t = ' '.join([t for t in text.split() if t not in russian_stop and t not in punctuations and '\n' != t and " " != t])
    t = [s for s in t if "\n" != s and '"' != s]
    text = "".join(t)
    text = re.sub(r"^\s+","",text)
    f.write("".join(text))
    f.write("\n")
f.close()

f = open("../tmp/text_for_fasttext.txt", "w")
for text in df["text_all"].values:
    text = text.lower()
    t = ' '.join([t for t in text.split() if '\n' != t and " " != t])
    t = [s for s in t if "\n" != s and '"' != s]
    text = "".join(t)
    text = re.sub(r"^\s+","",text)
    f.write("".join(text))
    f.write("\n")
f.close()

# Train fasttext model
import os
os.system("fasttext skipgram -input ../tmp/text_for_fasttext_without_stop.txt -output ../model/fasttext_model_without_stop -dim 100 -minCount 1")
os.system("fasttext skipgram -input ../tmp/text_for_fasttext.txt -output ../model/fasttext_model -dim 100 -minCount 1")

# create sentence vectors
import fastText
text = df["text_all"].values[:ntrain+ntest]
model = fastText.load_model("../model/fasttext_model_without_stop.bin")
sentence = np.zeros((text.shape[0], 100))
for i, t in enumerate(text):
    s = " ".join(t).replace("\n", " ")
    sentence[i, :] = model.get_sentence_vector(s)
train_out = pd.DataFrame(sentence[:ntrain,:], columns=["sentencevectors_without_stop_{}".format(i) for i in range(100)])
test_out = pd.DataFrame(sentence[ntrain:,:], columns=["sentencevectors_without_stop_{}".format(i) for i in range(100)])
to_parquet(train_out, "../vectors/sentence_vectors_without_stop_train.parquet")
to_parquet(test_out, "../vectors/sentence_vectors_without_stop_test.parquet")

model = fastText.load_model("../model/fasttext_model.bin")
sentence = np.zeros((text.shape[0], 100))
for i, t in enumerate(text):
    s = " ".join(t).replace("\n", " ")
    sentence[i, :] = model.get_sentence_vector(s)
train_out = pd.DataFrame(sentence[:ntrain,:], columns=["sentencevectors_{}".format(i) for i in range(100)])
test_out = pd.DataFrame(sentence[ntrain:,:], columns=["sentencevectors_{}".format(i) for i in range(100)])
to_parquet(train_out, "../vectors/sentence_vectors_train.parquet")
to_parquet(test_out, "../vectors/sentence_vectors_test.parquet")
