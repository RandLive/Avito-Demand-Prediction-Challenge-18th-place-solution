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
    cv_flg = False
    if is_read_data:
        X_train, X_test, y_train = read_train_test_data()
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y_train = read_parquet("../tmp/y_train.parquet").values.ravel()

    print("start cross validation")
    sub = pd.DataFrame()
    sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]

    print("X_train shape: ", X_train.shape)
    print("X_test shape: ", X_test.shape)

    features = X_train.columns
    # features = [col for col in X_train.columns if not col.startswith("oof_lgbm")]
    # X_train = X_train[features]
    # X_test = X_test[features]
    print("features: ", features)

    d_train = lgb.Dataset(X_train, label=y_train)
    del X_train; gc.collect()

    seed = np.random.randint(0,100000)
    # # default parameter
    # {'bagging_fraction': 0.9, 'verbose': 0, 'max_depth': 10, 'lambda_l
    # 2': 1.929748114943611, 'feature_fraction': 0.2235216889603703, 'objective': 'xentropy', 't
    # ask': 'train', 'lambda_l1': 4.590509375158124, 'num_leaves': 201, 'metric': {'rmse'}, 'boo
    # sting_type': 'gbdt', 'learning_rate': 0.02}
    if version==1:
        params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'xentropy',#'regression'
                'metric': 'rmse',#'xentropy'
                'num_leaves': 200,
                # 'min_sum_hessian_in_leaf': 10,
                'max_depth': 10,
                'learning_rate': 0.02,
                'feature_fraction': 0.2235, # 0.3
                'bagging_fraction': 0.9,
                # 'bagging_freq': 5,
                'lambda_l1': 10,
                'lambda_l2': 2,
                'verbose': 0,
                # 'feature_fraction_seed':seed,
                # 'bagging_seed':seed
                }
    elif version==2:
        print("MAX DROP 500 Drop DART.")
        params = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'xentropy',#'regression'
        'metric': 'rmse',#'xentropy'
        'num_leaves': 300,
        'max_drop':500,
        # 'min_data_in_leaf': 100,
        # 'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 1,
        'lambda_l2': 0.1,
        'verbose': 0,
        # 'feature_fraction_seed':seed,
        # 'bagging_seed':seed
        }

    elif version==3:
        print("Conservative model. GBDT")
        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'xentropy',#'regression'
        'metric': 'rmse',#'xentropy'
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'max_depth': 8,
        'learning_rate': 0.02,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'verbose': 0,
        # 'feature_fraction_seed':seed,
        # 'bagging_seed':seed
        }
    elif version==4:
        # DART
        params = {
                'task': 'train',
                'boosting_type': 'dart',
                'objective': 'xentropy',#'regression'
                'metric': 'rmse',#'xentropy'
                'num_leaves': 300,
                # 'min_data_in_leaf': 200,
                # 'max_depth': 10,
                'learning_rate': 0.05,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.9,
                # 'bagging_freq': 5,
                'lambda_l1': 0.5,
                'lambda_l2': 0.5,
                'verbose': 0,
                # 'feature_fraction_seed':seed,
                # 'bagging_seed':seed
                }

    print("======VERSION{}======".format(version))
    print("parameters...")
    print(params)
    if cv_flg:
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
        cv_scores = cvresult[len(cvresult)-1]
    else:
        bst = lgb.train(
                        params
                        , d_train
                        , 7390#2900
                        , verbose_eval=100
                        )
        cv_scores = "None"

    del d_train; gc.collect()
    y_pred = bst.predict(X_test)
    sub["deal_probability"] = bst.predict(X_test)
    sub["deal_probability"] = sub["deal_probability"].clip(0.0, 1.0)

    with open("../tmp/feature_importance_lgb.json", "w") as f:
        json.dump({f:g for f, g in zip(features, bst.feature_importance("gain"))}, f)

    # f, ax = plt.subplots(figsize=[50,80])
    # lgb.plot_importance(bst, max_num_features=50, ax=ax)
    # plt.title("Light GBM Feature Importance {}".format(cv_scores))
    # plt.savefig('feature_import_{}.png'.format(cv_scores))

    sub.to_csv("../output/lgb_valscore{}_version{}.csv".format(cv_scores, version), index=False)
    print("total cv score: ", cv_scores)
    notify_line("Done Training. CV Score {}".format(cv_scores))
    notify_line("Parameters: {}".format(str(params)))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
