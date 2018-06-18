import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
import json
import traceback
from bayes_opt import BayesianOptimization
from utils import *

try:
    for i in range(10):
        cv_flg = True
        seed = np.random.randint(0,10000)
        X_train, X_test, y_train = read_train_test_data_stacking()

        print("start cross validation")
        sub = pd.DataFrame()
        sub["item_id"] = pd.read_csv("../input/test.csv")["item_id"]

        print("X_train shape: ", X_train.shape)
        print("X_test shape: ", X_test.shape)

        features = X_train.columns
        print("features: ", features)

        d_train = lgb.Dataset(X_train, label=y_train)
        del X_train; gc.collect()

        # Parameter tuning
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
                'feature_fraction_seed':seed,
                'bagging_seed':seed,
                'verbose': 0

            }
            print("searching parameters...", params)
            cv_result = lgb.cv(params
                            , d_train
                            , 10000
                            , nfold=5
                            , verbose_eval=100
                            , early_stopping_rounds=200
                            , stratified=False
                            )
            num_round = len(cv_result["rmse-mean"])
            score = cv_result["rmse-mean"][num_round-1]
            return -1 * score

        seed = np.random.randint(0,100000)
        parameter_space = {
            'max_depth':(3,11),
            'learning_rate':(0.01,0.03),
            'num_leaves': (15,255),
            'feature_fraction': (0.5,1),
            'bagging_fraction': (0.3,1),
            'lambda_l1': (0.1,10),
            'lambda_l2': (0.1,10)
        }

        bayesopt = BayesianOptimization(func, parameter_space)
        bayesopt.maximize(init_points=30, n_iter=50)
        p = bayesopt.res['max']['max_params']


        bst_params = {
            'task': 'train',
            'max_depth':int(p["max_depth"]),
            "learning_rate":p["learning_rate"],
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'num_leaves': int(p["num_leaves"]),
            'feature_fraction': p["feature_fraction"],
            'bagging_fraction': p["bagging_fraction"],
            'lambda_l1': p["lambda_l1"],
            'lambda_l2': p["lambda_l2"],
            'feature_fraction_seed':seed,
            'bagging_seed':seed,
            'verbose': 0
        }

        print("Start CV...")
        cvresult = lgb.cv(
                        bst_params
                        , d_train
                        , 10000
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
        sub["deal_probability"] = bst.predict(X_test)
        sub["deal_probability"] = sub["deal_probability"].clip(0.0, 1.0)

        with open("../tmp/feature_importance_lgb_secondlayer.json", "w") as f:
            json.dump({f:g for f, g in zip(features, bst.feature_importance("gain"))}, f)

        sub.to_csv("../output/lgb_stacking_secondlayer_{}_{}.csv".format(cv_scores,seed), index=False)
        print("total cv score: ", cv_scores)
        notify_line("Done Training. CV Score {}".format(cv_scores))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
