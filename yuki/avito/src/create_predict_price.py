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

NFOLDS = 5
SEED = 42

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

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

try:
    print("\nData Load Stage")
    train = pd.read_csv('../input/train.csv', parse_dates = ["activation_date"]).drop(["deal_probability", "user_id"],axis=1)
    test = pd.read_csv('../input/test.csv', parse_dates = ["activation_date"]).drop("user_id", axis=1)
    df = pd.concat([train,test],axis=0)
    train = df[df.price==df.price]
    test = df[df.price!=df.price]
    y = train.price.values
    df.drop('price', axis=1, inplace=True)
    train_ids = train.item_id
    test_ids = test.item_id
    ntrain = train.shape[0]
    ntest = test.shape[0]
    df = pd.concat([train,test],axis=0)
    all_item_ids = df.item_id.tolist()
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    del train, test
    gc.collect()

    print("Feature Engineering")
    # NAN features
    nan_cols = ["description", "image", "param_1", "param_2", "param_3"]
    for cols in nan_cols:
        df[cols + "_is_NAN_bool"] = df[cols].fillna("MISSINGGGGGGGGGGGGGGGGG").apply(lambda x: int(x=="MISSINGGGGGGGGGGGGGGGGG"))
    df["num_NAN"] = df[[cols + "_is_NAN_bool" for cols in nan_cols]].sum(axis=1)

    df["image_top_1"].fillna(-999,inplace=True)

    print("\nCreate Time Variables")
    df["Weekday"] = df['activation_date'].dt.weekday
    df["Weekd of Year"] = df['activation_date'].dt.week
    df["Day of Month"] = df['activation_date'].dt.day

    # Create Validation Index and Remove Dead Variables
    df.drop(["activation_date","image"],axis=1,inplace=True)

    print("\nEncode Variables")
    categorical = ["region","city","parent_category_name","category_name","user_type","image_top_1"]
    print("Encoding :",categorical)

    # Encoder:
    lbl = preprocessing.LabelEncoder()
    for col in categorical:
        df[col] = lbl.fit_transform(df[col].astype(str))


    print("\nText Features")

    # Feature Engineering
    df['text_feat'] = df.apply(lambda row: ' '.join([
        str(row['param_1']),
        str(row['param_2']),
        str(row['param_3'])]),axis=1) # Group Param Features

    df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

    # Meta Text Features
    textfeats = ["description","text_feat", "title"]

    df['title'] = df['title'].apply(lambda x: cleanName(x))
    df["description"]   = df["description"].apply(lambda x: cleanName(x))
    for cols in textfeats:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
        df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
        # stop word, number, punctuations
        df[cols + '_num_stopwords'] = df[cols].apply(lambda x:len([w for w in x if w in russian_stop]))
        df[cols + '_num_punctuations'] = df[cols].apply(lambda x:len([w for w in x if w in punctuations]))
        df[cols + '_num_number'] = df[cols].apply(lambda x:len([w for w in x if w.isnumeric()]))
        # upper, lower


    print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")

    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        #"min_df":5,
        #"max_df":.9,
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
                #max_features=7000,
                preprocessor=get_col('text_feat'))),
            ('title',TfidfVectorizer(
                ngram_range=(1, 2),
                **tfidf_para,
                #max_features=7000,
                preprocessor=get_col('title')))
        ])

    start_vect=time.time()

    #Fit my vectorizer on the entire dataset instead of the training rows
    #Score improved by .0001
    vectorizer.fit(df.to_dict('records'))

    ready_df = vectorizer.transform(df.to_dict('records'))
    tfvocab = vectorizer.get_feature_names()
    print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

    # Drop Text Cols
    textfeats = ["description","text_feat", "title"]
    df.drop(textfeats, axis=1,inplace=True)

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    y = np.log(y + 0.0001)

    train_pred, test_pred = oof_sgd(ready_df[:ntrain], ready_df[ntrain:], y, "none", save=False)
    df['ridge_preds'] = np.concatenate([train_pred, test_pred])
    del train_pred, test_pred; gc.collect()

    # Combine Dense Features with Sparse Text Bag of Words Features
    X = hstack([csr_matrix(df.drop("item_id", axis=1).iloc[:ntrain,:].values),ready_df[:ntrain]]).tocsr()
    X_test = hstack([csr_matrix(df.drop("item_id", axis=1).iloc[ntrain:,:].values),ready_df[ntrain:]]).tocsr()
    tfvocab = df.columns.tolist() + tfvocab

    print("Feature Names Length: ",len(tfvocab))
    del df
    gc.collect();

    train_pred, test_pred = oof_lgbm(X, X_test, y, "none", save=False)
    train_pred = np.exp(train_pred) - 0.0001
    test_pred = np.exp(test_pred) - 0.0001
    df_out = pd.DataFrame()
    df_out["item_id"] = all_item_ids
    df_out["price_pred_1"] = np.concatenate([train_pred, test_pred])
    df_all = pd.concat([pd.read_csv("../input/train.csv", usecols=["item_id", "price"])
                        ,pd.read_csv("../input/test.csv", usecols=["item_id", "price"])])
    df_out = pd.merge(df_out, df_all, on="item_id", how="left")
    del df_all; gc.collect()
    df_out["diffprice_true_vs_pred"] = df_out["price_pred_1"] - df_out["price"]
    df_out.drop("price",axis=1,inplace=True)
    df_out = df_out.fillna(0)
    to_parquet(df_out, "../features/fe_item_price_pred_diff.parquet")
except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
