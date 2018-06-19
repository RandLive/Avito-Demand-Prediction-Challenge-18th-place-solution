import pandas as pd
import numpy as np
import lightgbm as lgb
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
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'xentropy',#'regression'
            'metric': 'rmse',#'xentropy'
            'num_leaves': 200,
            # 'min_sum_hessian_in_leaf': 10,
            'max_depth': 10,
            'learning_rate': 0.02,
            'feature_fraction': 0.4,
            'bagging_fraction': 0.9,
            # 'bagging_freq': 5,
            'lambda_l1': 3,
            'lambda_l2': 1,
            'verbose': 0,
            # 'feature_fraction_seed':seed,
            # 'bagging_seed':seed
            }
elif version=='single':
    X_train, X_test, y = read_train_test_data()
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
            'lambda_l2': 2,
            'verbose': 0,
            # 'feature_fraction_seed':seed,
            # 'bagging_seed':seed
            }

num_splits = 5
kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
val_predict = np.zeros(X_train.shape[0])
test_predict = np.zeros(X_test.shape[0])
X_train = X_train.values
X_test = X_test.values
for train_index, valid_index in kf.split(y):

    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]
    d_train = lgb.Dataset(X_train_fold, label=y_train_fold)
    d_eval = lgb.Dataset(X_valid_fold, label=y_valid_fold, reference=d_train)

    lgb_clf = lgb.train(
                        params
                        , d_train
                        , num_rounds
                        , d_eval
                        , verbose_eval=100
                        , early_stopping_rounds=200
                        )

    lgpred = lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration)
    y_pred = lgb_clf.predict(X_valid_fold, num_iteration=lgb_clf.best_iteration)
    val_predict[valid_index] = y_pred
    test_predict += lgpred / num_splits

val_predicts = pd.DataFrame(data=val_predict, columns=["deal_probability"])
test_predict = pd.DataFrame(data=test_predict, columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/lgb_oof_yuki_train.csv', index=False)
test_predict.to_csv('../output/lgb_oof_yuki_test.csv', index=False)
