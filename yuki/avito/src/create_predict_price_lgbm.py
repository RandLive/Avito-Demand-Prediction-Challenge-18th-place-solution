from utils import *

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
russian_stop = set(stopwords.words('russian'))
import string
punctuations = string.punctuation

# Viz
import re
import string
import time
import traceback
from utils import *

X_train, X_test, _ = read_train_test_data_all()
drop_cols = [f for f in X_train.columns if "oof_" in f or "price" in f]
X_train = X_train.drop(drop_cols, axis=1)
X_test = X_test.drop(drop_cols, axis=1)
y = pd.read_csv("../input/train.csv", usecols=["price"])["price"].values
y = np.log(y + 0.0001)

train_pred, test_pred = oof_lgbm(X_train, X_test, y, "none", save=False)
train_pred = np.exp(train_pred) - 0.0001
test_pred = np.exp(test_pred) - 0.0001
df_out = pd.DataFrame()
df_out["price_pred_2"] = np.concatenate([train_pred, test_pred])
df_all = pd.concat([pd.read_csv("../input/train.csv", usecols=["price"])
                    ,pd.read_csv("../input/test.csv", usecols=["price"])])

df_out["diffprice2_true_vs_pred"] = df_out["price_pred_2"] - df_out["price"]
df_out.drop("price",axis=1,inplace=True)
df_out = df_out.fillna(0)
to_parquet(df_out, "../stacking/oof_item_price_pred_diff_2.parquet")
