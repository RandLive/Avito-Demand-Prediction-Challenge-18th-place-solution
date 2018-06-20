import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from utils import *
import argparse


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

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

X_train, X_test, y = read_train_test_data()
X_train = X_train.fillna(0)#X.fillna(X.median()) # X.fillna(0)
X_train = X_train.replace(np.inf, 9999.999)
X_train = X_train.replace(-np.inf, -9999.999)
X_train = X_train.values
X_test = X_test.fillna(0)#X_test.fillna(X_test.median())
X_test = X_test.replace(np.inf, 9999.999)
X_test = X_test.replace(-np.inf, -9999.999)
X_test = X_test.values


num_splits = 5
kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
val_predict = np.zeros(X_train.shape[0])
test_predict = np.zeros(X_test.shape[0])
val_scores = []
for train_index, valid_index in kf.split(y):
    print("fold...")

    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]
    model = ExtraTreesRegressor(n_estimators=2000,
                                criterion="mse",
                                max_features=0.3,
                                max_depth=12,
                                min_samples_leaf=200,
                                min_impurity_decrease=0.5,
                                n_jobs=-1,
                                verbose=1
                                )

    model.fit(X_train_fold, y_train_fold)

    val_predict[valid_index] = model.predict(X_valid_fold)
    test_predict += model.predict(X_test) / num_splits
    val_scores.append(rmse(y_valid_fold, model.predict(X_valid_fold)))

validation_score = np.mean(val_scores)
val_predicts = pd.DataFrame(data=val_predict, columns=["deal_probability"])
test_predict = pd.DataFrame(data=test_predict, columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/et_oof_yuki_val{}_train.csv'.format(validation_score), index=False)
test_predict.to_csv('../output/et_oof_yuki_val{}_test.csv'.format(validation_score), index=False)
print("ExtraTree Val Score: {}".format(validation_score))
notify_line("ExtraTree Val Score: {}".format(validation_score))
