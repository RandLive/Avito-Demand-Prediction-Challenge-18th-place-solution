import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as xgb
import pyarrow as pa
import pyarrow.parquet as pq
import json
import traceback
from utils import *


def oof_xgb_classification(X_train, X_test, y_train, outputname):
    df_train_out = pd.DataFrame()
    df_test_out = pd.DataFrame()
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_train_pred': np.zeros(X_train.shape[0]),
        'y_test_pred': []
    }
    print("tuning parameters...")
    d_train = xgb.DMatrix(data["X_train"], label=data["y_train"])

    params = {
            # "booster"             : "dart",
            "objective"           : "reg:logistic",
            "eval_metric"         : "rmse",
            "eta"                 : 0.03,
            "max_depth"           : 10,
            # "min_child_weight"    : 1,
            # "gamma"               : 0.70,
            "subsample"           : 0.8,
            "colsample_bytree"    : 0.6,
            "alpha"               : 0.3,
            "lambda"              : 0.3,
            "silent"              : 1,
            # "seed"                : seed,
            # "max_delta_step"      : 1,
            # "scale_pos_weight"    : scale_weight
            }
    num_round = 10000
    print("cv...")
    cv_result = xgb.cv(params
                    , d_train
                    , num_round
                    , nfold=5
                    , verbose_eval=100
                    , early_stopping_rounds=100
                    , stratified=True
                    )
    bst_num_rounds = int(cvresult.shape[0])
    del d_train; gc.collect()

    n_splits = 5
    kfolds = get_kfolds()
    for ix_first, ix_second in tqdm(kfolds):
        d_train = xgb.DMatrix(data['X_train'][ix_first, :], label=data['y_train'][ix_first])
        print("start training")
        model = xgb.train(params
                        , d_train
                        , bst_num_rounds
                        , verbose_eval=100
                        )
        data['y_train_pred'][ix_second] = model.predict(xgb.DMatrix(data['X_train'][ix_second, :]))
        print("done pred val")
        data['y_test_pred'].append(model.predict(xgb.DMatrix(data['X_test'])))
        del d_train; gc.collect()

    data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

    df_train_out["oof_classification_{}_{}".format("xgb", outputname)] = data["y_train_pred"]
    df_test_out["oof_classification_{}_{}".format("xgb", outputname)] = data["y_test_pred"]
    to_parquet(df_train_out, "../stacking/oof_classification_{}_{}_train.parquet".format("xgb", outputname))
    to_parquet(df_test_out, "../stacking/oof_classification_{}_{}_test.parquet".format("xgb", outputname))


def oof_xgb_regression(X_train, X_test, y_train, target_idx, not_target_idx, outputname):
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    n_train_0 = len(not_target_idx)
    X_train_not = X_train[not_target_idx]
    X_train = X_train[target_idx]
    y_train = y_train[target_idx]
    X_test = np.concatenate([X_test, X_train_not])
    df_train_out = pd.DataFrame()
    df_test_out = pd.DataFrame()
    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_train_pred': np.zeros(X_train.shape[0]),
        'y_test_pred': []
    }
    print("tuning parameters...")
    d_train = xgb.DMatrix(data["X_train"], label=data["y_train"])

    params = {
            # "booster"             : "dart",
            "objective"           : "reg:logistic",
            "eval_metric"         : "rmse",
            "eta"                 : 0.03,
            "max_depth"           : 10,
            # "min_child_weight"    : 1,
            # "gamma"               : 0.70,
            "subsample"           : 0.8,
            "colsample_bytree"    : 0.6,
            "alpha"               : 0.3,
            "lambda"              : 0.3,
            "silent"              : 1,
            # "seed"                : seed,
            # "max_delta_step"      : 1,
            # "scale_pos_weight"    : scale_weight
            }
    num_round = 10000
    print("cv...")
    cv_result = xgb.cv(params
                    , d_train
                    , num_round
                    , nfold=5
                    , verbose_eval=100
                    , early_stopping_rounds=200
                    , stratified=True
                    )
    bst_num_rounds = len(cv_result["rmse-mean"])
    del d_train; gc.collect()

    n_splits = 5
    kfolds = get_kfolds()
    for ix_first, ix_second in tqdm(kfolds):
        d_train = xgb.DMatrix(data['X_train'][ix_first, :], label=data['y_train'][ix_first])
        print("start training")
        model = xgb.train(params
                        , d_train
                        , bst_num_rounds
                        , verbose_eval=100
                        )
        data['y_train_pred'][ix_second] = model.predict(xgb.DMatrix(data['X_train'][ix_second, :]))
        print("done pred val")
        data['y_test_pred'].append(model.predict(xgb.DMatrix(data['X_test'])))
        del d_train; gc.collect()

    data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
    train_out = np.zeros(n_train)
    test_out = np.zeros(n_test)

    test_out = data["y_test_pred"][:n_test]
    train_out[target_idx] = data["y_train_pred"]
    train_out[not_target_idx] = data["y_test_pred"][n_test:]
    df_train_out["oof_regression_{}_{}".format("xgb", outputname)] = train_out
    df_test_out["oof_regression_{}_{}".format("xgb", outputname)] = test_out
    to_parquet(df_train_out, "../stacking/oof_regression_{}_{}_train.parquet".format("xgb", outputname))
    to_parquet(df_test_out, "../stacking/oof_regression_{}_{}_test.parquet".format("xgb", outputname))

try:
    is_read_data = True

    if is_read_data:
        from utils import read_train_test_data#,tmp_read_train_valid
        X_train, X_test, y_train = read_train_test_data()
        # X_train, X_test, y_train = tmp_read_train_valid()
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y_train = read_parquet("../tmp/y_train.parquet").values.ravel()

    #  0/1 Classification
    y = (y_train==0).astype(int)
    oof_xgb_classification(X_train, X_test, y, "binary")

    #  0.55< Classification
    y = (y_train>=0.55).astype(int)
    oof_xgb_classification(X_train, X_test, y, "highclass")

    # 0 < y  regression
    target_idx = 0<y_train
    not_target_idx = ~target_idx
    oof_xgb_regression(X_train, X_test, y, target_idx, not_target_idx, "without_zero")

    # # 0 < y <= 0.55 regression
    # target_idx = (0<y_train) & (y_train<=0.55)
    # X = X_train[target_idx]
    # y = y_train[target_idx]
    # oof_xgb_regression(X, X_test, y, "midclass")
    #
    # # 0.55< y regression
    # target_idx = 0.55<y_train
    # X = X_train[target_idx]
    # y = y_train[target_idx]
    # oof_xgb_regression(X, X_test, y, "highclass")

    notify_line("Done.")

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
