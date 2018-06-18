import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
import json
import traceback
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

        d_train = xgb.DMatrix(X_train, label=y_train)
        del X_train; gc.collect()

        # Parameter tuning
        def func(max_depth, eta, subsample, colsample_bytree, alpha, lambda):
            params = {
                "objective": "reg:logistic",
                "eval_metric": "rmse",
                'max_depth':int(max_depth),
                "eta":eta,
                'colsample_bytree': colsample_bytree,
                'subsample': subsample,
                'alpha': alpha,
                'lambda': lambda,
                'seed':seed,
                'silent': 1
            }
            print("searching parameters...", params)
            cv_result = xgb.cv(params
                            , d_train
                            , 10000
                            , nfold=5
                            , verbose_eval=100
                            , early_stopping_rounds=200
                            , stratified=False
                            )['test-rmse-mean']
            num_rounds = int(cv_result.shape[0])
            score = cv_result.iloc[-1]
            return -1 * score

        seed = np.random.randint(0,100000)
        parameter_space = {
            'max_depth':(3,11),
            'eta':(0.01,0.03),
            'colsample_bytree': (0.5,1),
            'subsample': (0.3,1),
            'alpha': (0.1,10),
            'lambda': (0.1,10)
        }

        bayesopt = BayesianOptimization(func, parameter_space)
        bayesopt.maximize(init_points=50, n_iter=100)
        p = bayesopt.res['max']['max_params']

        bst_params = {
            "objective": "reg:logistic",
            'max_depth':int(p["max_depth"]),
            "eta":p["eta"],
            'eval_metric': 'rmse',
            'colsample_bytree': p["colsample_bytree"],
            'subsample': p["subsample"],
            'alpha': p["alpha"],
            'lambda': p["lambda"],
            'seed':seed,
            'verbose': 0
        }

        print("Start CV...")
        cvresult = xgb.cv(
                        bst_params
                        , d_train
                        , 10000
                        , early_stopping_rounds=200
                        , verbose_eval=100
                        , nfold=5
                        , stratified=False
                        )['test-rmse-mean']
        num_rounds = int(cvresult.shape[0])
        print("Done CV. best iteration: {}".format(num_rounds))
        watchlist = [(d_train, "train")]

        bst = xgb.train(
                        bst_params
                        , d_train
                        , num_rounds
                        , watchlist
                        , verbose_eval=100
                        )
        cv_scores = cvresult.iloc[-1]

        del d_train; gc.collect()
        d_test = xgb.DMatrix(X_test)
        y_pred = bst.predict(d_test)
        sub["deal_probability"] = bst.predict(d_test)
        sub["deal_probability"] = sub["deal_probability"].clip(0.0, 1.0)

        sub.to_csv("../output/xgb_stacking_secondlayer_{}_{}.csv".format(cv_scores,seed), index=False)
        print("total cv score: ", cv_scores)
        notify_line("Done Training. CV Score {}".format(cv_scores))

except:
    print(traceback.format_exc())
    notify_line(traceback.format_exc())
