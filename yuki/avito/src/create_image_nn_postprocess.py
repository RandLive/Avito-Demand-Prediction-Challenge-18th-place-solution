import numpy as np
import pandas as pd
import os
import gc
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import NMF
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import re
import string
from utils import *
from joblib import Parallel, delayed

y = pd.read_csv('../input/train.csv', usecols=["deal_probability"]).deal_probability.copy()

train = read_parquet("../tmp/fe_img_pretrained_nnmodel_train.parquet")
test = read_parquet("../tmp/fe_img_pretrained_nnmodel_test.parquet")
ntrain = train.shape[0]
ntest = test.shape[0]

df = pd.concat([train, test])

cols = ['Resnet50_label', 'Resnet50_score', 'xception_label', 'xception_score', 'Inception_label', 'Inception_score']
score_cols = ['Resnet50_score', 'xception_score', 'Inception_score']
cate_cols = ['Resnet50_label', 'xception_label', 'Inception_label']
df_out_train = pd.DataFrame()
df_out_test = pd.DataFrame()
for score_name, col in zip(score_cols, cate_cols):
    print("Creating features {}...".format(col))
    df_out_train[score_name] = train[score_name].values.astype(float)
    df_out_test[score_name] = test[score_name].values.astype(float)

    df_target = df[col].astype(str).fillna("NAN").values
    # basic features
    lbl = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder(sparse=True)
    vecs = ohe.fit_transform(lbl.fit_transform(df_target).reshape(df_target.shape[0], 1))

    # oof
    oof_sgd_train, oof_sgd_test = oof_sgd(vecs[:ntrain,:],vecs[ntrain:,:],y,"categorical_ohe_{}".format(col), save=False)
    df_out_train["oof_sgd_{}".format(col)] = oof_sgd_train
    df_out_test["oof_sgd_{}".format(col)] = oof_sgd_test
    df_out_train["oof_sgd_{}_interaction".format(col)] = oof_sgd_train * df_out_train[score_name].values
    df_out_test["oof_sgd_{}_interaction".format(col)] = oof_sgd_test * df_out_test[score_name].values
    print("lgbm oof...")
    oof_lgbm_train, oof_lgbm_test = oof_lgbm(vecs[:ntrain,:],vecs[ntrain:,:],y,"categorical_ohe_{}".format(col), save=False)
    df_out_train["oof_lgbm_{}".format(col)] = oof_lgbm_train
    df_out_test["oof_lgbm_{}".format(col)] = oof_lgbm_test
    df_out_train["oof_lgbm_{}_interaction".format(col)] = oof_lgbm_train * df_out_train[score_name].values
    df_out_test["oof_lgbm_{}_interaction".format(col)] = oof_lgbm_test * df_out_test[score_name].values

to_parquet(df_out_train, "../features/oof_img_pretrained_nnmodel_train.parquet")
to_parquet(df_out_test, "../features/oof_img_pretrained_nnmodel_test.parquet")
