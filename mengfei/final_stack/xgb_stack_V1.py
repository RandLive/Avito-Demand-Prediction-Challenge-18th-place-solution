# -*- coding: utf-8 -*-
"""
by Mengfei Li
"""
import pandas as pd
import xgboost as xgb
import numpy as np
import gc
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.externals import joblib


debug = False
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])
else:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:10000]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv", nrows=1000, parse_dates = ["activation_date"])
print("loading data done!")

id_train = train_df["item_id"]
id_test = test_df["item_id"]
len_train = len(train_df)
len_test = len(test_df)

def Do_Label_Enc(df):
      print("label encoding ...")
      lbl = LabelEncoder()
      cat_col = ["region", "city", "parent_category_name", 
                 "category_name", "user_type", "image_top_1",
                 "param_1", "param_2", "param_3",
                 ]
      for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()

def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

full_df = pd.concat([train_df, test_df], sort=False)
Do_Label_Enc(full_df)

# NN Steeve
print("load nn oof ...")
nn_oof_train = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_train.csv')
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_test.csv')
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'nn_oof_1'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
del nn_oof_train, nn_oof_test
gc.collect()

# LGB Yuki
print("load yuki_lgb oof ...")
yuki_lgb_train = pd.read_csv('../input/lgb_oof_yuki_train.csv')
yuki_lgb_train.drop("user_id", axis=1, inplace=True)
yuki_lgb_test = pd.read_csv('../input/mlp_oof_yuki_test.csv')
yuki_lgb_test.drop("user_id", axis=1, inplace=True)
yuki_lgb_full = pd.concat([yuki_lgb_train, yuki_lgb_test], sort=False)
yuki_lgb_full.rename(columns={'deal_probability': 'yuki_lgb_1'}, inplace=True)
full_df = pd.merge(full_df, yuki_lgb_full, on='item_id', how='left')
del yuki_lgb_train, yuki_lgb_test
gc.collect()

# FM huiqing
print("load huiqin_fm oof ...")
huiqin_fm_train = pd.read_csv('../input/wordbatch_fmtrl_submission_train.csv')
huiqin_fm_train.drop("user_id", axis=1, inplace=True)
huiqin_fm_test = pd.read_csv('../input/wordbatch_fmtrl_submissionV3_5fold.csv')
#huiqin_fm_test.drop("user_id", axis=1, inplace=True)
huiqin_fm_full = pd.concat([huiqin_fm_train, huiqin_fm_test], sort=False)
huiqin_fm_full.rename(columns={'deal_probability': 'huiqin_fm_1'}, inplace=True)
full_df = pd.merge(full_df, huiqin_fm_full, on='item_id', how='left')
del huiqin_fm_train, huiqin_fm_test
gc.collect()


full_df.drop(["user_id", "description",
              "activation_date", "title",
              "image", "deal_probability",
              "item_id"], axis=1, inplace=True)
            
train_df = full_df[:len_train]
test_df = full_df[len_train:]


params = {'eta': 0.02,  # 0.03,
          "booster": "gbtree",
          "tree_method": "gpu_hist",
          'max_depth': 6,  # 18
          "nthread": 3,
          'subsample': 0.9,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 0.8,
          'min_child_weight': 10,
          'alpha': 1,
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'random_state': 42,
          'silent': True,
#          'grow_policy': 'lossguide',       
          # "nrounds": 8000
          }

numIter = 0
rmse_sum = 0
folds=5
kf = KFold(n_splits=folds, random_state=42, shuffle=True)

Dtest = xgb.DMatrix(test_df)
for train_index, valid_index in kf.split(y):      
      numIter += 1      
      print("training in fold " + str(numIter))
      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
      
      Dtrain = xgb.DMatrix(X_train, y_train)
      Dvalid = xgb.DMatrix(X_valid, y_valid)
      
      watchlist = [(Dtrain, 'train'), (Dvalid, 'valid')]
      model = xgb.train(params, Dtrain, 30000, watchlist, maximize=False, early_stopping_rounds=200,
                              verbose_eval=100)
     
      print("save model ...")
      joblib.dump(model, "xgb_stack_{}.pkl".format(numIter))
#      load model
#      lgb_clf = joblib.load("lgb.pkl")

      print("Model Evaluation Stage")
      rmse_vl = rmse(y_valid, model.predict(Dvalid,ntree_limit=model.best_ntree_limit))
      print( "RMSE:", rmse_vl )
      
      xgbpred = model.predict(Dtest,ntree_limit=model.best_ntree_limit)
            
      xgbsub = pd.DataFrame(xgbpred,columns=["deal_probability"],index=id_test)
      xgbsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
      xgbsub.to_csv("stacked_xgb_5fold_sub_{}.csv".format(numIter),index=True,header=True)
      
      rmse_sum += rmse_vl
                  
      del X_train, X_valid, y_train, y_valid, Dtrain, Dvalid
      gc.collect()
      
print("mean rmse is:", rmse_sum/folds)