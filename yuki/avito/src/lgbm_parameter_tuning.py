import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import dill
import pyarrow.parquet as pq
import json
import traceback
from utils import *
from bayes_opt import BayesianOptimization
seed = np.random.randint(0,100000)

def func(max_depth, learning_rate, num_leaves, feature_fraction, bagging_fraction, lambda_l1, lambda_l2):
    params = {
        'task': 'train',
        'max_depth':int(max_depth),
        "learning_rate":learning_rate,
        'boosting_type': 'gbdt',
        'objective': 'xentropy',
        'metric': {'rmse'},
        'num_leaves': int(num_leaves),
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'verbose': 0

    }
    print("searching parameters...", params)
    cv_result = lgb.cv(params
                    , d_train
                    , 10000
                    , nfold=5
                    , verbose_eval=100
                    # , feval=eval_fscore
                    , early_stopping_rounds=200
                    , stratified=False
                    )
    num_round = len(cv_result["rmse-mean"])
    score = cv_result["rmse-mean"][num_round-1]
    return -1 * score

# Read input data

X_train, X_test, y_train = read_train_test_data()
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
d_train = lgb.Dataset(X_train, label=y_train)

# tuning parameter

parameter_space = {
    'max_depth':(8,11),
    'learning_rate':(0.02,0.05),
    'num_leaves': (127,255),
    'feature_fraction': (0.2,0.7),
    'bagging_fraction': (0.2,0.9),
    'lambda_l1': (0.5,20),
    'lambda_l2': (0.5,20)
}

bayesopt = BayesianOptimization(func, parameter_space)
bayesopt.maximize(init_points=50, n_iter=100)
p = bayesopt.res['max']['max_params']


bst_params = {
    'task': 'train',
    'max_depth':int(p["max_depth"]),
    "learning_rate":p["learning_rate"],
    'boosting_type': 'gbdt',
    'objective': 'xentropy',
    'metric': {'rmse'},
    'num_leaves': int(p["num_leaves"]),
    'feature_fraction': p["feature_fraction"],
    'bagging_fraction': p["bagging_fraction"],
    'lambda_l1': p["lambda_l1"],
    'lambda_l2': p["lambda_l2"],
    'verbose': 0
}
print("best parameters: ", bst_params)

with open("../tmp/lgb_best_params_stackmodel.dat", "wb") as f:
    dill.dump(bst_params, f)

cvresult = lgb.cv(
                bst_params
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
                bst_params
                , d_train
                , num_rounds
                , verbose_eval=100
                )
cv_scores = cvresult[len(cvresult)-1]

del d_train; gc.collect()
y_pred = bst.predict(X_test)
sub = pd.DataFrame()
sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]
sub["deal_probability"] = bst.predict(X_test)
sub["deal_probability"] = sub["deal_probability"].clip(0.0, 1.0)

sub.to_csv("../output/lgb_valscore{}_stackmodel.csv".format(cv_scores), index=False)
print("total cv score: ", cv_scores)
notify_line("Done Training. CV Score {}".format(cv_scores))
notify_line("Parameters: {}".format(str(params)))
