import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from utils import *
import argparse

# specify the version.
parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', default='ensemble', help='version')
args = parser.parse_args()
version = args.version
is_read_data = True

train = pd.read_csv('../input/train.csv', usecols=['user_id', 'item_id'])
test = pd.read_csv('../input/test.csv', usecols=['user_id', 'item_id'])
train_user_ids = train.user_id.tolist()
train_item_ids = train.item_id.tolist()
test_user_ids = test.user_id.tolist()
test_item_ids = test.item_id.tolist()

if version=='ensemble':
    if is_read_data:
        X_train, X_test, y = read_train_test_data()
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y = read_parquet("../tmp/y_train.parquet").values.ravel()
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

    num_rounds = 2000
    params = {
    # "objective"           : "reg:linear",
    "objective"           : "reg:logistic",
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

            }
elif version=='single':
    X_train, X_test, y = read_train_test_data()
    num_rounds = 10000
    params = {
    # "objective"           : "reg:linear",
    "objective"           : "reg:logistic",
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
            }

num_splits = 5
kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
val_predict = np.zeros(X_train.shape[0])
test_predict = np.zeros(X_test.shape[0])
X_train = X_train.values
X_test = X_test.values
d_test = xgb.DMatrix(X_test)
for train_index, valid_index in kf.split(y):

    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]
    d_train = xgb.DMatrix(X_train_fold, label=y_train_fold)
    d_eval = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
    watchlist = [(d_train, 'train'), (d_eval, 'valid')]

    xgb_clf = xgb.train(
                        params
                        , d_train
                        , num_rounds
                        , watchlist
                        , verbose_eval=100
                        , early_stopping_rounds=200
                        )

    xgpred = xgb_clf.predict(d_test)
    y_pred = xgb_clf.predict(d_eval)
    val_predict[valid_index] = y_pred
    test_predict += xgpred / num_splits

val_predicts = pd.DataFrame(data=val_predict, columns=["deal_probability"])
test_predict = pd.DataFrame(data=test_predict, columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/xgb_oof_yuki_train.csv', index=False)
test_predict.to_csv('../output/xgb_oof_yuki_test.csv', index=False)
