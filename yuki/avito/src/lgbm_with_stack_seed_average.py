import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
import json
import traceback
from utils import *
import argparse

# specify the version.
parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', default=1, help='version')
args = parser.parse_args()
version = int(args.version)
num_average = 10

try:
    sub = pd.DataFrame()
    sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]
    sub["deal_probability"] = 0
    X_train = read_parquet("../tmp/X_train.parquet")
    X_test = read_parquet("../tmp/X_test.parquet")
    y_train = read_parquet("../tmp/y_train.parquet").values.ravel()
    X_tr_sta, X_te_sta, _ = read_train_test_data_stacking()
    X_train = pd.concat([X_train, X_tr_sta],axis=1)
    X_test = pd.concat([X_test, X_te_sta],axis=1)
    del X_tr_sta, X_te_sta; gc.collect()
    nogain_features =[]
    f =open("../tmp/no_gain_features_stack_version.txt")
    for l in f.readlines():
        nogain_features.append(l.replace("\n",""))
        f.close()
    drop_cols = [col for col in X_train.columns if col in nogain_features]
    X_train = X_train.drop(drop_cols, axis=1)
    X_test = X_test.drop(drop_cols, axis=1)

    print("start cross validation")
    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    features = X_train.columns
    print("features: ", features)

    d_train = lgb.Dataset(X_train, label=y_train)
    del X_train; gc.collect()

    for i in range(num_average):
        print("ITERATION {}".format(i))

        seed = np.random.randint(0,100000)

        params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'xentropy',#'regression'
                'metric': 'rmse',#'xentropy'
                'num_leaves': 200,
                # 'min_sum_hessian_in_leaf': 10,
                'max_depth': 10,
                'learning_rate': 0.02,
                'feature_fraction': 0.2235,
                'bagging_fraction': 0.9,
                # 'bagging_freq': 5,
                'lambda_l1': 10,
                'lambda_l2': 2,
                'verbose': 0,
                'feature_fraction_seed':seed,
                'bagging_seed':seed,
                'seed':seed
                }

        print("parameters...")
        print(params)

        print("Start CV...")
        cvresult = lgb.cv(
                        params
                        , d_train
                        , 20000
                        , early_stopping_rounds=200
                        , verbose_eval=100
                        , nfold=5
                        , stratified=False
                        )['rmse-mean']
        num_rounds = int(len(cvresult))
        print("Done CV. best iteration: {}".format(num_rounds))

        bst = lgb.train(
                        params
                        , d_train
                        , num_rounds
                        , verbose_eval=100
                        )

        sub["deal_probability"] += bst.predict(X_test) / num_average

    sub["deal_probability"] = sub["deal_probability"].clip(0.0, 1.0)
    sub.to_csv("../output/lgb_with_stack_seed_average_{}.csv".format(num_average), index=False)
    notify_line("Seed Average Done:")

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
