import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from utils import *
russian_stop = set(stopwords.words('russian'))
stop_2 = set([w for w in open("../tmp/russian_stopwords.txt", "r").readlines()])
russian_stop = russian_stop.union(stop_2)
import string
punctuations = string.punctuation
russian_stop = russian_stop.union(set(punctuations))
from joblib import Parallel, delayed
import gc

train = pd.read_csv('../input/train.csv', usecols=["item_id","description","title"])
test = pd.read_csv('../input/test.csv',usecols=["item_id","description","title"])

n_train = train.shape[0]
n_test = test.shape[0]

df = pd.concat([train, test])
del train, test; gc.collect()

def stemming_text(text):
    stemmer = SnowballStemmer("russian")
    out = []
    for t in text.split():
        stemed = stemmer.stem(t)
        if stemed not in russian_stop:
            out.append(stemed)
    return " ".join(out)

df["description"] = Parallel(n_jobs=-1,verbose=1)([delayed(stemming_text)(x) for x in df.description.fillna("")])
df["title"] = Parallel(n_jobs=-1,verbose=1)([delayed(stemming_text)(x) for x in df.title.fillna("")])
df.iloc[:n_train,:].to_csv("../input/train_stemmed.csv", index=False)
df.iloc[n_train:,:].to_csv("../input/test_stemmed.csv", index=False)
