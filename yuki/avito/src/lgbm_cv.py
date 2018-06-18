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
    is_read_data = True
    cv_flg = True
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
    sub["oof_stacking_level1_lgbm_seed42_{}".format(version)] = 0
    val_pred = np.zeros(X_train.shape[0])

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    features = X_train.columns
    # features = [col for col in X_train.columns if not col.startswith("oof_lgbm")]
    # X_train = X_train[features]
    # X_test = X_test[features]
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
                'lambda_l2': 1,
                'verbose': 0,
                'feature_fraction_seed':seed,
                'bagging_seed':seed
                }
        num_rounds = 7390
        print("parameters...")
        print(params)

        bst = lgb.train(
                        params
                        , d_train
                        , num_rounds
                        , d_eval
                        , verbose_eval=100
                        , early_stopping_rounds=200
                        )


        del d_train; gc.collect()
        sub["oof_stacking_level1_lgbm_seed42_{}".format(version)] += bst.predict(X_test) / 5
        y_valid_pred = bst.predict(X_valid_fold)
        val_pred[ix_valid] = y_valid_pred
        cv_scores.append(rmse(y_valid_fold, y_valid_pred))

    cv_score = np.mean(cv_scores)
    train_oof = pd.DataFrame(val_pred, columns=["oof_stacking_level1_lgbm_seed42_{}".format(version)])
    sub = sub.drop("item_id", axis=1)
    to_parquet(train_oof, "../stacking/oof_stacking_level1_lgbm_seed42_{}_{}_train.parquet".format(version, seed))
    to_parquet(sub, "../stacking/oof_stacking_level1_lgbm_seed42_{}_{}_test.parquet".format(version, seed))

    # For Error Analysis
    # error_analysis = pd.read_csv("../input/train.csv")
    # error_analysis["deal_probability_pred"] = val_pred
    # error_analysis["rmse"] = rmse(error_analysis["deal_probability"].values, val_pred)
    # error_analysis.to_csv("../tmp/error_analysis_lgbm.csv", index=False)
    notify_line("Done. CV Score: {}".format(cv_score))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
