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
parser.add_argument('--obj', '-o', default="xentropy", help='objective')
args = parser.parse_args()
obj = args.obj
version = int(args.version)


try:
    is_read_data = True
    cv_flg = True
    if is_read_data:
        X_train, X_test, y_train = read_train_test_data_all()
        oof_cols = [f for f in X_train.columns if "oof_" in f]
        X_train = X_train.drop(oof_cols, axis=1)
        X_test = X_test.drop(oof_cols, axis=1)
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y_train = read_parquet("../tmp/y_train.parquet").values.ravel()


    print("start cross validation")
    sub = pd.DataFrame()
    sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]
    sub["oof_stacking_level1_lgbm_no_oof_{}_{}".format(obj, version)] = 0
    val_pred = np.zeros(X_train.shape[0])

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    features = X_train.columns
    print("features: ", features)

    with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)

    def rmse(y, y0):
        assert len(y) == len(y0)
        return np.sqrt(np.mean(np.power((y - y0), 2)))

    cv_scores = []
    for ix_train, ix_valid in kfolds:

        X_train_fold = X_train.iloc[ix_train,:]
        X_valid_fold = X_train.iloc[ix_valid,:]
        y_train_fold = y_train[ix_train]
        y_valid_fold = y_train[ix_valid]
        d_train = lgb.Dataset(X_train_fold, label=y_train_fold)
        d_eval = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=d_train)

        seed = np.random.randint(0,100000)
        num_rounds = 10000
        params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': obj,#'regression'
                'metric': 'rmse',#'xentropy'
                'num_leaves': 300,
                # 'min_sum_hessian_in_leaf': 10,
                # 'max_depth': 10,
                'learning_rate': 0.03,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.9,
                # 'bagging_freq': 5,
                'lambda_l1': 1,
                'lambda_l2': 0.1,
                'verbose': 0,
                'feature_fraction_seed':seed,
                'bagging_seed':seed
                }
        print("parameters...")
        print(params)

        bst = lgb.train(
                        params
                        , d_train
                        , 10000
                        , d_eval
                        , verbose_eval=100
                        , early_stopping_rounds=200
                        )


        del d_train; gc.collect()
        sub["oof_stacking_level1_lgbm_no_oof_{}_{}".format(obj, version)] += bst.predict(X_test) / 5
        y_valid_pred = bst.predict(X_valid_fold)
        val_pred[ix_valid] = y_valid_pred
        cv_scores.append(rmse(y_valid_fold, y_valid_pred))

    cv_score = np.mean(cv_scores)
    train_oof = pd.DataFrame(val_pred, columns=["oof_stacking_level1_lgbm_no_oof_{}_{}".format(obj,version)])
    sub = sub.drop("item_id", axis=1)
    to_parquet(train_oof, "../stacking/oof_stacking_level1_lgbm_no_oof_{}_{}_{}_train.parquet".format(obj, version, seed))
    to_parquet(sub, "../stacking/oof_stacking_level1_lgbm_no_oof_{}_{}_{}_test.parquet".format(obj,version, seed))

    notify_line("Done. CV Score: {}".format(cv_score))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
