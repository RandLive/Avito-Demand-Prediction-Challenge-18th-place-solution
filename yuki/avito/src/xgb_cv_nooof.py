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

try:
    cv_flg = True
    X_train, X_test, y_train = read_train_test_data_all()
    dropcols = [col for col in X_train.columns if "oof_" in col]
    X_train = X_train.drop(dropcols, axis=1)
    X_test = X_test.drop(dropcols, axis=1)

    print("start cross validation")
    sub = pd.DataFrame()
    sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]
    sub["oof_stacking_level1_xgb_nooof_seed42_{}".format(version)] = 0
    val_pred = np.zeros(X_train.shape[0])

    print("X_train shape: ", X_train.shape)

    features = X_train.columns
    print("features: ", features)

    with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)

    def rmse(y, y0):
        assert len(y) == len(y0)
        return np.sqrt(np.mean(np.power((y - y0), 2)))

    cv_scores = []
    d_test = xgb.DMatrix(X_test)
    del X_test; gc.collect()
    for ix_train, ix_valid in kfolds:
        X_train_fold = X_train.iloc[ix_train,:]
        X_valid_fold = X_train.iloc[ix_valid,:]
        y_train_fold = y_train[ix_train]
        y_valid_fold = y_train[ix_valid]

        d_train = xgb.DMatrix(X_train_fold, label=y_train_fold)
        del X_train_fold
        d_val = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
        del X_valid_fold
        watchlist = [(d_train, 'train'), (d_val, 'valid')]
        # scale_weight = 1#len(y[y==0]) / len(y[y==1])
        seed = np.random.randint(0,100000)
        params = {
        # "booster"             : "dart",
        "objective"           : "reg:logistic",
        "eval_metric"         : "rmse",
        "eta"                 : 0.02,
        "max_depth"           : 9,
        # "min_child_weight"    : 1,
        # "gamma"               : 0.70,
        "subsample"           : 0.8,
        "colsample_bytree"    : 0.5,
        "alpha"               : 10,
        "lambda"              : 1,
        "silent"              : 1,
        "seed"                : seed,
        # "max_delta_step"      : 1,
        # "scale_pos_weight"    : scale_weight
        }
        num_rounds = 10000


        print("parameters...")
        print(params)

        bst = xgb.train(
                        params
                        , d_train
                        , num_rounds
                        , watchlist
                        , verbose_eval=100
                        , early_stopping_rounds=200
                        )

        del d_train; gc.collect()
        sub["oof_stacking_level1_xgb_nooof_seed42_{}".format(version)] += bst.predict(d_test) / 5
        y_valid_pred = bst.predict(d_val)
        val_pred[ix_valid] = y_valid_pred
        cv_scores.append(rmse(y_valid_fold, y_valid_pred))

    cv_score = np.mean(cv_scores)
    train_oof = pd.DataFrame(val_pred, columns=["oof_stacking_level1_xgb_nooof_seed42_{}".format(version)])
    sub = sub.drop("item_id", axis=1)
    to_parquet(train_oof, "../stacking/oof_stacking_level1_xgb_nooof_seed42_{}_train.parquet".format(version))
    to_parquet(sub, "../stacking/oof_stacking_level1_xgb_nooof_seed42_{}_test.parquet".format(version))
    notify_line("XGB CV Done. SV Score: {}".format(cv_score))


except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
