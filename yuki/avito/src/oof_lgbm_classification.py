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

parser = argparse.ArgumentParser()
parser.add_argument('--feat', '-o', default="", help='feature type')
args = parser.parse_args()
feat = args.feat

def oof_lgbm_classification(X_train, X_test, y_train, outputname, seed=0):
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
    d_train = lgb.Dataset(data["X_train"], label=data["y_train"])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        # 'min_data_in_leaf':200,
        'max_depth': 10,
        'num_leaves': 280,
        'learning_rate': 0.02,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'verbose': 0,
        'lambda_l1':3,
        'lambda_l2':1,
        'feature_fraction_seed':seed,
        'bagging_seed':seed
    }
    num_round = 10000
    print("cv...")
    cv_result = lgb.cv(params
                    , d_train
                    , num_round
                    , nfold=5
                    , verbose_eval=100
                    , early_stopping_rounds=200
                    , stratified=True
                    )
    bst_num_rounds = len(cv_result["auc-mean"])
    print("Done CV for classification. best iteration: {}".format(bst_num_rounds))
    del d_train; gc.collect()

    n_splits = 5
    kfolds = get_kfolds()
    for ix_first, ix_second in tqdm(kfolds):
        d_train = lgb.Dataset(data['X_train'][ix_first, :], label=data['y_train'][ix_first])
        print("start training")
        model = lgb.train(params
                        , d_train
                        , bst_num_rounds
                        , verbose_eval=100
                        )
        data['y_train_pred'][ix_second] = model.predict(data['X_train'][ix_second, :])
        print("done pred val")
        data['y_test_pred'].append(model.predict(data['X_test']))
        del d_train; gc.collect()

    data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)

    df_train_out["oof_classification_{}_{}".format("lgbm", outputname)] = data["y_train_pred"]
    df_test_out["oof_classification_{}_{}".format("lgbm", outputname)] = data["y_test_pred"]
    to_parquet(df_train_out, "../stacking/oof_classification_{}_{}_{}_train.parquet".format("lgbm", outputname, seed))
    to_parquet(df_test_out, "../stacking/oof_classification_{}_{}_{}_test.parquet".format("lgbm", outputname, seed))


# def oof_lgbm_regression(X_train, X_test, y_train, target_idx, not_target_idx, outputname):
#     n_test = X_test.shape[0]
#     n_train = X_train.shape[0]
#     n_train_0 = len(not_target_idx)
#     X_train_not = X_train[not_target_idx]
#     X_train = X_train[target_idx]
#     y_train = y_train[target_idx]
#     X_test = np.concatenate([X_test, X_train_not])
#     df_train_out = pd.DataFrame()
#     df_test_out = pd.DataFrame()
#     data = {
#         'X_train': X_train,
#         'y_train': y_train,
#         'X_test': X_test,
#         'y_train_pred': np.zeros(X_train.shape[0]),
#         'y_test_pred': []
#     }
#     print("tuning parameters...")
#     d_train = lgb.Dataset(data["X_train"], label=data["y_train"])
#
#     params = {
#         'task': 'train',
#         'boosting_type': 'gbdt',
#         'objective': 'regression',
#         'metric': 'rmse',
#         # 'min_data_in_leaf':200,
#         # 'max_depth': 15,
#         'num_leaves': 300,
#         'learning_rate': 0.02,
#         'feature_fraction': 0.3,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 3,
#         'verbose': 0,
#         'lambda_l1':5,
#         'lambda_l2':1
#     }
#     num_round = 10000
#     print("cv...")
#     cv_result = lgb.cv(params
#                     , d_train
#                     , num_round
#                     , nfold=5
#                     , verbose_eval=100
#                     , early_stopping_rounds=100
#                     , stratified=True
#                     )
#     bst_num_rounds = len(cv_result["rmse-mean"])
#     print("Done CV for regression classification. best iteration: {}".format(bst_num_rounds))
#     del d_train; gc.collect()
#
#     n_splits = 5
#     kfolds = get_kfolds()
#     for ix_first, ix_second in tqdm(kfolds):
#         d_train = lgb.Dataset(data['X_train'][ix_first, :], label=data['y_train'][ix_first])
#         print("start training")
#         model = lgb.train(params
#                         , d_train
#                         , bst_num_rounds
#                         , verbose_eval=100
#                         )
#         data['y_train_pred'][ix_second] = model.predict(data['X_train'][ix_second, :])
#         print("done pred val")
#         data['y_test_pred'].append(model.predict(data['X_test']))
#         del d_train; gc.collect()
#
#     data['y_test_pred'] = np.array(data['y_test_pred']).T.mean(axis=1)
#     train_out = np.zeros(n_train)
#     test_out = np.zeros(n_test)
#
#     test_out = data["y_test_pred"][:n_test]
#     train_out[target_idx] = data["y_train_pred"]
#     train_out[not_target_idx] = data["y_test_pred"][n_test:]
#     df_train_out["oof_regression_{}_{}".format("lgbm", outputname)] = train_out
#     df_test_out["oof_regression_{}_{}".format("lgbm", outputname)] = test_out
#     to_parquet(df_train_out, "../stacking/oof_regression_{}_{}_train.parquet".format("lgbm", outputname))
#     to_parquet(df_test_out, "../stacking/oof_regression_{}_{}_test.parquet".format("lgbm", outputname))

try:
    for i in range(1):

        is_read_data = True
        if is_read_data:
            from utils import read_train_test_data#,tmp_read_train_valid
            X_train, X_test, y_train = read_train_test_data()
            if feat=="select200":
                nogain_features =[]
                f =open("../tmp/no_gain_features_selection_model.txt")
                for l in f.readlines():
                    nogain_features.append(l.replace("\n",""))
                f.close()
                drop_cols = [col for col in X_train.columns if col in nogain_features]
            elif feat=="nooof":
                drop_cols = [col for col in X_train.columns if "oof_" in col]
            else:
                drop_cols = []
            X_train = X_train.drop(drop_cols, axis=1)
            X_test = X_test.drop(drop_cols, axis=1)
        else:
            X_train = read_parquet("../tmp/X_train.parquet")
            X_test = read_parquet("../tmp/X_test.parquet")
            y_train = read_parquet("../tmp/y_train.parquet").values.ravel()

        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train
        seed = np.random.randint(0,10000)
        #  0/1 Classification
        # y = (y_train==0).astype(int)
        # oof_lgbm_classification(X_train, X_test, y, "binary_{}".format(feat), seed=seed)

        #  0.55< Classification
        y = (y_train>=0.55).astype(int)
        oof_lgbm_classification(X_train, X_test, y, "highclass_{}".format(feat), seed=seed)

        # # 0 < y  regression
        # target_idx = 0<y_train
        # not_target_idx = ~target_idx
        # oof_lgbm_regression(X_train, X_test, y, target_idx, not_target_idx, "without_zero")


        notify_line("Done.")

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
