from nltk.corpus import stopwords 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from math import sqrt

debug = False
print("loading data ...")
if debug == False:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    del train_df["deal_probability"]; gc.collect()
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])
else:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:50000]
    y = train_df["deal_probability"]
    del train_df["deal_probability"]; gc.collect()
    test_df = pd.read_csv("../input/test.csv",  nrows=1000, parse_dates = ["activation_date"])


def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())


def feature_engineering(df):
    # All the feature engineering here
    
    def Do_Text_Hash(df):
        print("feature engineering -> hash text ...")
        df['text_feat'] = df.apply(lambda row: ' '.join([str(row['param_1']),
          str(row['param_2']), str(row['param_3'])]),axis=1)
        df['title_description'] = df.apply(lambda row: ' '.join([str(row['title']),
          str(row['description'])]),axis=1)
        df['text_feat_title_description'] = df.apply(lambda row: ' '.join([str(row['text_feat']),
          str(row['title_description'])]),axis=1)
    
        
    def Do_Datetime(df):
        print("feature engineering -> date time ...")
        df["wday"] = df["activation_date"].dt.weekday.astype('int8')
        df["week"] = df["activation_date"].dt.week.astype('int16')
        df["dom"] = df["activation_date"].dt.day.astype('int16')
        
    def Do_Label_Enc(df):
        print("feature engineering -> lable encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               "param_1", "param_2", "param_3"]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
    
    def Do_NA(df):
        print("feature engineering -> fill na ...")
        df["price"] = np.log(df["price"]+0.001).astype('float32')
        df["price"].fillna(-999,inplace=True)
        df["image_top_1"].fillna(-999,inplace=True)
        df["image"].fillna("noinformation",inplace=True)
        df["description"].fillna("нетинформации",inplace=True)
             
    def Do_Drop(df):
        df.drop('activation_date', axis=1, inplace=True)
     
    # choose which functions to run 
    Do_Text_Hash(df)    
    Do_Datetime(df)
    Do_Label_Enc(df)
    Do_NA(df)
    
    
    Do_Drop(df)
    
    gc.collect()
    return df




feature_engineering(train_df)



