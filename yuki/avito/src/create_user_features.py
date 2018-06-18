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
import seaborn as sns
import matplotlib.pyplot as plt
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
df["dayofweek"] = df.activation_date.dt.weekday
training["dayofweek"] = training.activation_date.dt.weekday
testing["dayofweek"] = testing.activation_date.dt.weekday
df["one"] = 1

del train_active, test_active
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

def merge_df_by_userid(df1, df2):
    return pd.merge(df1, df2, on="user_id", how="left")

user_ids = df.user_id.unique()
df_out = pd.DataFrame()
df_out["user_id"] = user_ids
# User X Price
print("User X Price Features...")
user_group = df[["user_id", "price"]].dropna().groupby("user_id")
df_out = merge_df_by_userid(df_out, user_group["price"].median().reset_index().rename(columns={"price":"user_price_median"}))
df_out = merge_df_by_userid(df_out, user_group["price"].mean().reset_index().rename(columns={"price":"user_price_mean"}))
df_out = merge_df_by_userid(df_out, user_group["price"].std().reset_index().rename(columns={"price":"user_price_std"}))
df_out = merge_df_by_userid(df_out, user_group["price"].min().reset_index().rename(columns={"price":"user_price_min"}))
df_out = merge_df_by_userid(df_out, user_group["price"].max().reset_index().rename(columns={"price":"user_price_max"}))

to_parquet(df_out, "../features/fe_user_price_base.parquet")
del df_out; gc.collect()

# User period
print("User period features...")
df_period = pd.DataFrame()
df_period["user_count"] = df.groupby("user_id")["one"].sum()
df_period["latest_date"] = df.groupby("user_id")["activation_date"].max()
df_period["first_date"] = df.groupby("user_id")["activation_date"].min()
df_period["user_period"] = (df_period["latest_date"] - df_period["first_date"]).dt.days
df_period["user_average_period"] = df_period["user_period"] / df_period["user_count"]
df_period.drop(["latest_date", "first_date"], axis=1, inplace=True)
to_parquet(df_period.reset_index(), "../features/fe_user_period_base.parquet")
del df_period; gc.collect()

# key: [User,dayofweek], agg:count, price stats
df_out = pd.DataFrame()
df_out["user_weekday_count"] = df.groupby(["user_id", "dayofweek"])["one"].sum()
df_out["user_weekday_price_median"] = df.groupby(["user_id", "dayofweek"])["price"].median()
df_out["user_weekday_price_std"] = df.groupby(["user_id", "dayofweek"])["price"].std()
df_out["user_weekday_price_max"] = df.groupby(["user_id", "dayofweek"])["price"].max()
df_out["user_weekday_price_min"] = df.groupby(["user_id", "dayofweek"])["price"].min()
df_out = df_out.reset_index()
keys = ["user_id", "dayofweek"]
df_train_out = pd.merge(training[keys], df_out, on=keys, how="left").drop(keys, axis=1)
df_test_out = pd.merge(testing[keys], df_out, on=keys, how="left").drop(keys, axis=1)
to_parquet(df_train_out, "../features/fe_weekday_X_user_features_train.parquet")
to_parquet(df_test_out, "../features/fe_weekday_X_user_features_test.parquet")

# User X categorical count
print("Creating User X categorical Features...")
df_out = pd.DataFrame()
df_out["user_id"] = user_ids
cate_cols = ["parent_category_name","category_name", "param_1", "param_2", "param_3", "city", "region"]
user_group = df[["user_id"]+cate_cols].fillna("_NAN_").groupby("user_id")
df_out = merge_df_by_userid(df_out, user_group[cate_cols].agg(pd.Series.nunique)\
                    .reset_index().rename(columns={
                    "parent_category_name":"user_parent_category_name_count"
                    ,"category_name":"user_category_name_count"
                    ,"param_1":"user_param_1_count"
                    ,"param_2":"user_param_2_count"
                    ,"param_3":"user_param_3_count"
                    ,"city":"user_city_count"
                    ,"region":"user_region_count"
                    }))

del user_group;gc.collect()

cate_cols = ["image_top_1"]
train_test = pd.concat([training, testing])
user_group = train_test[["user_id"]+cate_cols].fillna("_NAN_").groupby("user_id")
df_out = merge_df_by_userid(df_out, user_group[cate_cols].agg(pd.Series.nunique)\
                    .reset_index().rename(columns={
                    "image_top_1":"user_image_top_1_count"
                    }))

del user_group, train_test;gc.collect()
to_parquet(df_out, "../features/fe_user_categorical_count_features.parquet")



#  User X item_seq_number
print("Creating User X item_seq_number Features...")
df_out = pd.DataFrame()
df_out["user_id"] = user_ids
user_group = df[["user_id", "item_seq_number"]].groupby("user_id")
df_out = merge_df_by_userid(df_out,user_group["item_seq_number"].max().reset_index().rename(columns={"item_seq_number":"user_item_seq_number_max"}))
df_out = merge_df_by_userid(df_out,user_group["item_seq_number"].min().reset_index().rename(columns={"item_seq_number":"user_item_seq_number_min"}))
to_parquet(df_out, "../features/fe_user_item_seq_number.parquet")

# Aggregated Features. inspired by
# https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/code
df_out = pd.DataFrame()
df_out["user_id"] = user_ids

used_cols = ['item_id', 'user_id']

train = pd.read_csv('../input/train.csv', usecols=used_cols)
train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test = pd.read_csv('../input/test.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['date_from', 'date_to'])

all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
all_samples.drop_duplicates(['item_id'], inplace=True)

del train_active
del test_active
gc.collect()

all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

all_periods['days_up'] = (all_periods['date_to'] - all_periods['date_from']).dt.days

# gp = all_periods.groupby(['item_id'])[['days_up']]
# gp_df = pd.DataFrame()
# gp_df['days_up_sum'] = gp.sum()['days_up']
# gp_df['days_up_max'] = gp.max()['days_up']
# gp_df['days_up_min'] = gp.min()['days_up']
# gp_df['days_up_mean'] = gp.mean()['days_up']
# gp_df['times_put_up'] = gp.count()['days_up']
# gp_df.reset_index(inplace=True)
# gp_df.rename(index=str, columns={'index': 'item_id'})
# to_parquet(gp_df, "../features/fe_item_aggregated.parquet")

all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods = all_periods.merge(gp_df, on='item_id', how='left')

all_periods = all_periods.merge(all_samples, on='item_id', how='left')

gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
    .rename(index=str, columns={
        'days_up_sum': 'avg_days_up_sum_user',
        'days_up_max': 'avg_days_up_max_user',
        'days_up_min': 'avg_days_up_min_user',
        'days_up_mean': 'avg_days_up_mean_user',
        'times_put_up': 'avg_times_up_user'
    })
gp2 = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].max().reset_index() \
    .rename(index=str, columns={
        'days_up_sum': 'max_days_up_user',
        'days_up_max': 'max_days_up_max_user',
        'days_up_min': 'max_days_up_min_user',
        'days_up_mean': 'max_days_up_mean_user',
        'times_put_up': 'max_times_up_user'
    })
gp = merge_df_by_userid(gp, gp2)
gp2 = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].min().reset_index() \
    .rename(index=str, columns={
        'days_up_sum': 'min_days_up_user',
        'days_up_max': 'min_days_up_max_user',
        'days_up_min': 'min_days_up_min_user',
        'days_up_mean': 'min_days_up_mean_user',
        'times_put_up': 'min_times_up_user'
    })
gp = merge_df_by_userid(gp, gp2)
del gp2; gc.collect()

n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
    .rename(index=str, columns={
        'item_id': 'n_user_items'
    })
gp = gp.merge(n_user_items, on='user_id', how='left')
to_parquet(gp, "../features/fe_user_aggregated.parquet")
