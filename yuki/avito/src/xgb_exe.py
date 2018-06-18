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
    is_read_data = True
    if is_read_data:
        from utils import read_train_test_data#,tmp_read_train_valid
        X_train, X_test, y_train = read_train_test_data()
        # X_train, X_test, y_train = tmp_read_train_valid()
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y_train = read_parquet("../tmp/y_train.parquet").values.ravel()

    print("start cross validation")
    sub = pd.DataFrame()
    sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]

    print("X_train shape: ", X_train.shape)

    features = X_train.columns
    print("features: ", features)

    d_train = xgb.DMatrix(X_train, label=y_train)
    del X_train
    d_test = xgb.DMatrix(X_test)
    watchlist = [(d_train, "train")]
    # scale_weight = 1#len(y[y==0]) / len(y[y==1])
    seed = np.random.randint(0,100000)
    if version==1:
        # Base model
        params = {
                # "booster"             : "dart",
                "objective"           : "reg:linear",
                # "objective"           : "reg:logistic",
                "eval_metric"         : "rmse",
                "eta"                 : 0.03,
                "max_depth"           : 8,
                # "min_child_weight"    : 1,
                # "gamma"               : 0.70,
                "subsample"           : 0.8,
                "colsample_bytree"    : 0.2,
                "alpha"               : 5,
                "lambda"              : 1,
                "silent"              : 1,
                # "seed"                : seed,
                # "max_delta_step"      : 1,
                # "scale_pos_weight"    : scale_weight
                }
    elif version==2:
        # No overfitting
        params = {
                # "booster"             : "dart",
                "objective"           : "reg:logistic",
                "eval_metric"         : "rmse",
                "eta"                 : 0.02,
                "max_depth"           : 8,
                "subsample"           : 0.8,
                "colsample_bytree"    : 0.2,
                "alpha"               : 8,
                "lambda"              : 3,
                "silent"              : 1,
                # "seed"                : seed,
                # "max_delta_step"      : 1,
                # "scale_pos_weight"    : scale_weight
                }
        num_rounds = 7000 # approximately
    elif version==3:
        # No Overfitting
        params = {
                # "booster"             : "dart",
                "objective"           : "reg:logistic",
                "eval_metric"         : "rmse",
                "eta"                 : 0.02,
                "max_depth"           : 8,
                # "min_child_weight"    : 1,
                # "gamma"               : 0.70,
                "subsample"           : 0.8,
                "colsample_bytree"    : 0.2,
                "alpha"               : 4,
                "lambda"              : 4,
                "silent"              : 1,
                # "seed"                : seed,
                # "max_delta_step"      : 1,
                # "scale_pos_weight"    : scale_weight
                }
    print("======VERSION{}======".format(version))
    print("parameters...")
    print(params)
    if cv_flg:
        cvresult = xgb.cv(
                        params
                        , d_train
                        , num_boost_round=10000
                        , early_stopping_rounds=200
                        , verbose_eval=100
                        , nfold=5
                        , stratified=False
                        )['test-rmse-mean']
        num_rounds = int(cvresult.shape[0])
        print("CV Done. Best Iteration: ", num_rounds)

        bst = xgb.train(
                        params
                        , d_train
                        , num_rounds
                        , watchlist
                        , verbose_eval=100
                        )
        cv_scores = cvresult.iloc[-1]
    else:
        bst = xgb.train(
                        params
                        , d_train
                        , 2000
                        , watchlist
                        , verbose_eval=100
                        )

        cv_scores = "None"

    sub["deal_probability"] = bst.predict(d_test)

    sub.to_csv("../output/xgb_valscore{}_version{}.csv".format(cv_scores, version), index=False)
    print("total cv score: ", cv_scores)
    notify_line("Done Training. CV Score {}".format(cv_scores))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
