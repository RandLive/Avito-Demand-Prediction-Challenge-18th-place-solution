# -*- coding: utf-8 -*-
"""
by: Mengfei Li
"""
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from catboost import CatBoostRegressor


debug = True


mode = 'none'

print("loading data ...")
used_cols = ["item_id", "user_id"]

if debug == False:
      nrows_train = 1503424
      nrows_test = 508438
else:
      nrows_train = 100000
      nrows_test = 1000
      
train_df = pd.read_csv("../input/train.csv", nrows=nrows_train, parse_dates = ["activation_date"])
y = train_df["deal_probability"]
test_df = pd.read_csv("../input/test.csv", nrows=nrows_test, parse_dates = ["activation_date"])

print("loading data done!")

id_train = train_df["item_id"]
id_test = test_df["item_id"]
len_train = len(train_df)
len_test = len(test_df)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

def Do_Label_Enc(df):
      print("label encoding ...")
      lbl = LabelEncoder()
      cat_col = ["region", "city", "parent_category_name", 
                 "category_name", "user_type", "image_top_1",
                 "param_1", "param_2", "param_3", "image", "user_id",
                 ]
      for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()

def feature_Eng_On_Price_Make_More_Cat(df):
    print('feature engineering -> on price and SEQ ...')    
    df["price"] = np.log(df["price"]+0.001).astype("float32") 
    df["price"].fillna(-1,inplace=True) 
    df["price+"] = np.round(df["price"]*3.5).astype(np.int16) # 4.8
    df["item_seq_number+"] = np.round(df["item_seq_number"]/100).astype(np.int16)

def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_oof(clf, x_train, y, x_test):
            
    oof_train = np.zeros((len_train,))
    oof_test = np.zeros((len_test,))
    oof_test_skf = np.empty((NFOLDS, len_test))

    for i, (train_index, test_index) in enumerate(kf):
#        print('Ridege oof Fold {}'.format(i))
        x_tr = x_train[train_index]       
        y = np.array(y)
        y_tr = y[train_index]
        x_te = x_train[test_index]      
        clf.train(x_tr, y_tr)       
        oof_train[test_index] = clf.predict(x_te)        
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/region_income.csv", sep=";", names=["region", "income"])
train_df = train_df.merge(tmp, on="region", how="left")
test_df = test_df.merge(tmp, on="region", how="left")
del tmp; gc.collect()
# =============================================================================
# Add population
# =============================================================================
tmp = pd.read_csv("../input/city_population_wiki_v3.csv")
train_df = train_df.merge(tmp, on="city", how="left")
test_df = test_df.merge(tmp, on="city", how="left")
del tmp; gc.collect()

# =============================================================================
# new image_feature
# =============================================================================
tmp = pd.read_csv("../input/train_image_keypoints.csv")
train_df = train_df.merge(tmp, on="image", how="left")
tmp = pd.read_csv("../input/test_image_keypoints.csv")
test_df = test_df.merge(tmp, on="image", how="left")
del tmp; gc.collect()

# =============================================================================
# concat df
# =============================================================================
full_df = pd.concat([train_df, test_df], sort=False)
Do_Label_Enc(full_df)
feature_Eng_On_Price_Make_More_Cat(full_df)

#full_df["region_city"] = full_df["region"]*full_df["city"]
# new hash

full_df.drop([
              "user_id", 
              "description",
              "activation_date", 
              "title",
              "image", 
              "deal_probability",
#              "item_id",            
              "region", 
              "city", 
              "parent_category_name",
              "category_name", 
              "item_seq_number",
              "image_top_1", 
              "price",
              "price+",
              "user_type",
              "item_seq_number+",
              "param_1", 
              "param_2", 
              "param_3",
              ], axis=1, inplace=True)
full_df.fillna(-1234, inplace=True)


# all features
print("load yuki_catboost oof ...")
all_features_df = pd.read_parquet('../input/all_features.parquet', engine='pyarrow')
all_features_df = all_features_df.iloc[:nrows_train+nrows_test]
#full_df["ridge_preds_1"] = all_features_df.ridge_preds_1.values
full_df["ridge_preds_2"] = all_features_df.ridge_preds_2.values

# RMSE: 0.21045107551893638
# =============================================================================
# load oofs
# =============================================================================
# LGB Yuki
print("load yuki stack oof ...")
yuki_lgb_train = pd.read_csv('../input/lgb_stacking_10folds_valscore0.21006420577444712_yuki_train.csv',nrows=nrows_train)
#yuki_lgb_train.drop("user_id", axis=1, inplace=True)
yuki_lgb_test = pd.read_csv('../input/lgb_stacking_10folds_valscore0.21006420577444712_yuki_test.csv', nrows=nrows_test)
#yuki_lgb_test.drop("user_id", axis=1, inplace=True)
yuki_lgb_full = pd.concat([yuki_lgb_train, yuki_lgb_test], sort=False)
yuki_lgb_full.rename(columns={'deal_probability': 'yuki_stack_final'}, inplace=True)
full_df = pd.merge(full_df, yuki_lgb_full, on='item_id', how='left')
full_df["yuki_stack_final"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_train, yuki_lgb_test
gc.collect()


# lstm Yuki
print("load yuki stack oof ...")
yuki_lgb_train = pd.read_csv('../input/oof_rnn_ver3_lstm_embed2_train_oof.csv',nrows=nrows_train)
yuki_lgb_train.drop("user_id", axis=1, inplace=True)
yuki_lgb_test = pd.read_csv('../input/oof_rnn_ver3_lstm_embed2_test.csv', nrows=nrows_test)
yuki_lgb_test.drop("user_id", axis=1, inplace=True)
yuki_lgb_full = pd.concat([yuki_lgb_train, yuki_lgb_test], sort=False)
yuki_lgb_full.rename(columns={'deal_probability': 'yuki_lstm'}, inplace=True)
full_df = pd.merge(full_df, yuki_lgb_full, on='item_id', how='left')
full_df["yuki_lstm"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_train, yuki_lgb_test
gc.collect()

# =============================================================================
# Last day adding
# =============================================================================
print("load last 1 ...")
nn_oof_train = pd.read_csv('../input1/capsule_novec_V1_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input1/capsule_novec_V1_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'last_1'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["last_1"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()

print("load last 2 ...")
nn_oof_train = pd.read_csv('../input1/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_deslda_tilda_mlp_b128_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input1/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_deslda_tilda_mlp_b128_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'last_2'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["last_2"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()

print("load last 3 ...")
nn_oof_train = pd.read_csv('../input1/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_vgg_b128_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input1/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_vgg_b128_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'last_3'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["last_3"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


print("load last 4 ...")
nn_oof_train = pd.read_csv('../input1/svd_5fold_train_oof.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input1/svd_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'last_4'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["last_4"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


print("load last 5 ...")
nn_oof_train = pd.read_csv('../input1/v100_emb_all_80_itseqcat_price_p4_pr1_catn_p_rm_deslog_ict_pc_dll_dnl_rgb_apw_newmm_descon_ndam_b128_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input1/v100_emb_all_80_itseqcat_price_p4_pr1_catn_p_rm_deslog_ict_pc_dll_dnl_rgb_apw_newmm_descon_ndam_b128_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'last_5'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["last_5"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()



# =============================================================================
# 
# =============================================================================
# NN Steeve
print("load nn oof ...")
nn_oof_train = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'nn_oof_1'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["nn_oof_1"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()

# NN Steeve
print("load nn oof 5 ...")
nn_oof_train = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_b128_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_b128_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'nn_oof_5'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["nn_oof_5"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()

## NN Steeve
print("load nn oof 4 ...")
nn_oof_train = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_b128_img_dim200_incepv3_avgpool_imagemiddle_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_b128_img_dim200_incepv3_avgpool_imagemiddle_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'nn_oof_4'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["nn_oof_4"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()



## NN Steeve
print("load nn oof huang 3 ...")
nn_oof_train = pd.read_csv('../input/mercari_no2_sol_emb_50_price_deslen_nn_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/mercari_no2_sol_emb_50_price_deslen_nn_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'nn_oof_3'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["nn_oof_3"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()

### zhuang DNN
print("load dcnn oof zhuang ...")
nn_oof_train = pd.read_csv('../input/dpcnn_5fold_train_oof.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/dpcnn_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'dcnn_oof'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["dcnn_oof"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


### zhuang NN aver pool
print("load aver pool oof zhuang ...")
nn_oof_train = pd.read_csv('../input/avgpool_5fold_train_oof.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/avgpool_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'aver_pool_oof'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["aver_pool_oof"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


### zhuang DNN
print("load mlp oof zhuang ...")
nn_oof_train = pd.read_csv('../input/mlp_5fold_train_oof.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/mlp_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'mlp_zhuang_oof'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["mlp_zhuang_oof"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


### huang v100 1
print("load v100 2 oof huang ...")
nn_oof_train = pd.read_csv('../input/v100_emb_all_80_itseqcat_price_p4_pr1_catn_p_rm_deslog_ict_pc_dll_dnl_rgb_apw_newmm_b128_5fold_train.csv',nrows=nrows_train)
nn_oof_train.drop("user_id", axis=1, inplace=True)
nn_oof_test = pd.read_csv('../input/v100_emb_all_80_itseqcat_price_p4_pr1_catn_p_rm_deslog_ict_pc_dll_dnl_rgb_apw_newmm_b128_5fold_test.csv', nrows=nrows_test)
nn_oof_full = pd.concat([nn_oof_train, nn_oof_test], sort=False)
nn_oof_full.rename(columns={'deal_probability': 'v100_oof'}, inplace=True)
full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
full_df["v100_oof"].clip(0.0, 1.0, inplace=True)
del nn_oof_train, nn_oof_test
gc.collect()


# LGB Yuki
print("load yuki_lgb_2 oof ...")
yuki_lgb_train = pd.read_csv('../input/lgb_oof_yuki_train_1.csv',nrows=nrows_train)
yuki_lgb_train.drop("user_id", axis=1, inplace=True)
yuki_lgb_test = pd.read_csv('../input/lgb_oof_yuki_test_1.csv', nrows=nrows_test)
yuki_lgb_test.drop("user_id", axis=1, inplace=True)
yuki_lgb_full = pd.concat([yuki_lgb_train, yuki_lgb_test], sort=False)
yuki_lgb_full.rename(columns={'deal_probability': 'yuki_lgb_2'}, inplace=True)
full_df = pd.merge(full_df, yuki_lgb_full, on='item_id', how='left')
full_df["yuki_lgb_2"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_train, yuki_lgb_test
gc.collect()


# LGB Yuki
print("load yuki_lgb v tfid oof ...")
yuki_lgb_train = pd.read_csv('../input/lgb_various_tfidf_oof_yuki_train.csv',nrows=nrows_train)
yuki_lgb_train.drop("user_id", axis=1, inplace=True)
yuki_lgb_test = pd.read_csv('../input/lgb_various_tfidf_oof_yuki_test.csv', nrows=nrows_test)
yuki_lgb_test.drop("user_id", axis=1, inplace=True)
yuki_lgb_full = pd.concat([yuki_lgb_train, yuki_lgb_test], sort=False)
yuki_lgb_full.rename(columns={'deal_probability': 'yuki_lgb_v'}, inplace=True)
full_df = pd.merge(full_df, yuki_lgb_full, on='item_id', how='left')
full_df["yuki_lgb_v"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_train, yuki_lgb_test
gc.collect()


# cat Yuki
print("load yuki_cat_2 oof ...")
yuki_cat_2_train = pd.read_csv('../input/catboost_oof_yuki_val0.21733364922240023_train.csv',nrows=nrows_train)
yuki_cat_2_train.drop("user_id", axis=1, inplace=True)
yuki_cat_2_test = pd.read_csv('../input/catboost_oof_yuki_val0.21733364922240023_test.csv', nrows=nrows_test)
yuki_cat_2_test.drop("user_id", axis=1, inplace=True)
yuki_cat_2_full = pd.concat([yuki_cat_2_train, yuki_cat_2_test], sort=False)
yuki_cat_2_full.rename(columns={'deal_probability': 'yuki_cat_2'}, inplace=True)
full_df = pd.merge(full_df, yuki_cat_2_full, on='item_id', how='left')
full_df["yuki_cat_2"].clip(0.0, 1.0, inplace=True)
del yuki_cat_2_train, yuki_cat_2_test
gc.collect()


# cat Yuki
#print("load ridge 2 oof ...")
#tmp = pd.read_csv('../input/ridge_oof.csv',nrows=(nrows_train+nrows_test))
#full_df["ridge_oof"] = tmp.ridge_preds.values
#gc.collect()


# xgb finnal
# mean rmse is: 0.21289664867629216
#print("load xgb final 2 oof ML ...")
#tmp = pd.read_csv('../input/xgb_final_oof_1.csv',nrows=(nrows_train+nrows_test))
#full_df["xgb_final_oof_1"] = tmp.xgb_here_2.values
#gc.collect()


# xgb final mengfei
#print("load mengfei final stack oof ...")
#yuki_catboost_train = pd.read_csv('../input/stacked_xgb_5fold_sub_all_oof.csv')
#yuki_catboost_train = yuki_catboost_train[:nrows_train]
#yuki_catboost_test = pd.read_csv('../input/stacked_xgb_5fold_sub_all.csv')
#yuki_catboost_test = yuki_catboost_test[:nrows_test]
#yuki_catboost_full = pd.concat([yuki_catboost_train, yuki_catboost_test], sort=False)
#yuki_catboost_full.rename(columns={'deal_probability':'ml_xgb_final'}, inplace=True)
#full_df["ml_xgb_final"] = np.array(yuki_catboost_full["ml_xgb_final"])
#full_df["ml_xgb_final"].clip(0.0, 1.0, inplace=True)
#del yuki_catboost_train, yuki_catboost_test
#gc.collect()



# catboost huiqin
print("load final ml oof ...")
yuki_catboost_train = pd.read_parquet('../input/catboost_submissionV2_train.parquet', engine='pyarrow')
yuki_catboost_train = yuki_catboost_train[:nrows_train]
yuki_catboost_test = pd.read_parquet('../input/catboost_submissionV2_test.parquet', engine='pyarrow')
yuki_catboost_test = yuki_catboost_test[:nrows_test]
yuki_catboost_full = pd.concat([yuki_catboost_train, yuki_catboost_test], sort=False)
yuki_catboost_full.rename(columns={'catboost_submissionV2': 'yuki_catboost_1'}, inplace=True)
full_df["yuki_catboost_1"] = np.array(yuki_catboost_full["yuki_catboost_1"])
full_df["yuki_catboost_1"].clip(0.0, 1.0, inplace=True)
del yuki_catboost_train, yuki_catboost_test
gc.collect()



print("load final ml oof ...")
yuki_catboost_train = pd.read_csv('../input/catboost_submissionV4_train.csv', engine='pyarrow')
yuki_catboost_train = yuki_catboost_train[:nrows_train]
yuki_catboost_test = pd.read_csv('../input/catboost_submissionV4_test.csv', engine='pyarrow')
yuki_catboost_test = yuki_catboost_test[:nrows_test]
yuki_catboost_full = pd.concat([yuki_catboost_train, yuki_catboost_test], sort=False)
yuki_catboost_full.rename(columns={'catboost_submissionV2': 'yuki_catboost_4'}, inplace=True)
full_df["yuki_catboost_4"] = np.array(yuki_catboost_full["yuki_catboost_4"])
full_df["yuki_catboost_1"].clip(0.0, 1.0, inplace=True)
del yuki_catboost_train, yuki_catboost_test
gc.collect()


# lstm huiqin
print("load yuki_lstm oof ...")
yuki_lstm_train = pd.read_parquet('../input/oof_rnn_ver2_lstm_embed1_train.parquet', engine='pyarrow')
yuki_lstm_train = yuki_lstm_train[:nrows_train]
yuki_lstm_test = pd.read_parquet('../input/oof_rnn_ver2_lstm_embed1_test.parquet', engine='pyarrow')
yuki_lstm_test = yuki_lstm_test[:nrows_test]
yuki_lstm_full = pd.concat([yuki_lstm_train, yuki_lstm_test], sort=False)
yuki_lstm_full.rename(columns={'oof_rnn_ver2_lstm_embed1': 'yuki_lstm_1'}, inplace=True)
full_df["yuki_lstm_1"] = np.array(yuki_lstm_full["yuki_lstm_1"])
full_df["yuki_lstm_1"].clip(0.0, 1.0, inplace=True)
del yuki_lstm_train, yuki_lstm_test
gc.collect()


# lgb binary yuki
print("load yuki_lgb_binary oof ...")
yuki_lgb_binary_train = pd.read_parquet('../input/oof_classification_lgbm_binary_nooof_9817_train.parquet', engine='pyarrow')
yuki_lgb_binary_train = yuki_lgb_binary_train[:nrows_train]
yuki_lgb_binary_test = pd.read_parquet('../input/oof_classification_lgbm_binary_nooof_9817_test.parquet', engine='pyarrow')
yuki_lgb_binary_test = yuki_lgb_binary_test[:nrows_test]
yuki_lgb_binary_full = pd.concat([yuki_lgb_binary_train, yuki_lgb_binary_test], sort=False)
yuki_lgb_binary_full.rename(columns={'oof_classification_lgbm_binary_nooof': 'yuki_lgb_binary_1'}, inplace=True)
full_df["yuki_lgb_binary_1"] = np.array(yuki_lgb_binary_full["yuki_lgb_binary_1"])
full_df["yuki_lgb_binary_1"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_binary_train, yuki_lgb_binary_test
gc.collect()



# lgb binary yuki
print("load yuki_lgb_binary 2 oof ...")
yuki_lgb_binary_train = pd.read_parquet('../input/oof_classification_lgbm_binary_normal_6364_train.parquet', engine='pyarrow')
yuki_lgb_binary_train = yuki_lgb_binary_train[:nrows_train]
yuki_lgb_binary_test = pd.read_parquet('../input/oof_classification_lgbm_binary_normal_6364_test.parquet', engine='pyarrow')
yuki_lgb_binary_test = yuki_lgb_binary_test[:nrows_test]
yuki_lgb_binary_full = pd.concat([yuki_lgb_binary_train, yuki_lgb_binary_test], sort=False)
yuki_lgb_binary_full.rename(columns={'oof_classification_lgbm_binary_normal': 'yuki_lgb_binary_2'}, inplace=True)
full_df["yuki_lgb_binary_2"] = np.array(yuki_lgb_binary_full["yuki_lgb_binary_2"])
full_df["yuki_lgb_binary_2"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_binary_train, yuki_lgb_binary_test
gc.collect()


# lgb binary yuki
print("load yuki_lgb_binary 3 oof ...")
yuki_lgb_binary_train = pd.read_parquet('../input/oof_stacking_level1_mlp_2_base_binary_crossentropy_train.parquet', engine='pyarrow')
yuki_lgb_binary_train = yuki_lgb_binary_train[:nrows_train]
yuki_lgb_binary_test = pd.read_parquet('../input/oof_stacking_level1_mlp_2_base_binary_crossentropy_test.parquet', engine='pyarrow')
yuki_lgb_binary_test = yuki_lgb_binary_test[:nrows_test]
yuki_lgb_binary_full = pd.concat([yuki_lgb_binary_train, yuki_lgb_binary_test], sort=False)
yuki_lgb_binary_full.rename(columns={'oof_stacking_level1_mlp_2_base_binary_crossentropy': 'yuki_lgb_binary_3'}, inplace=True)
full_df["yuki_lgb_binary_3"] = np.array(yuki_lgb_binary_full["yuki_lgb_binary_3"])
full_df["yuki_lgb_binary_3"].clip(0.0, 1.0, inplace=True)
del yuki_lgb_binary_train, yuki_lgb_binary_test
gc.collect()


# lgb binary yuki
print("load yuki_lgbm_highclass oof ...")
yuki_lgbm_highclass_train = pd.read_parquet('../input/oof_classification_lgbm_highclass_6982_train.parquet', engine='pyarrow')
yuki_lgbm_highclass_train = yuki_lgbm_highclass_train[:nrows_train]
yuki_lgbm_highclass_test = pd.read_parquet('../input/oof_classification_lgbm_highclass_6982_test.parquet', engine='pyarrow')
yuki_lgbm_highclass_test = yuki_lgbm_highclass_test[:nrows_test]
yuki_lgbm_highclass_full = pd.concat([yuki_lgbm_highclass_train, yuki_lgbm_highclass_test], sort=False)
yuki_lgbm_highclass_full.rename(columns={'oof_classification_lgbm_highclass': 'yuki_lgbm_highclass_1'}, inplace=True)
full_df["yuki_lgbm_highclass_1"] = np.array(yuki_lgbm_highclass_full["yuki_lgbm_highclass_1"])
full_df["yuki_lgbm_highclass_1"].clip(0.0, 1.0, inplace=True)
del yuki_lgbm_highclass_train, yuki_lgbm_highclass_test
gc.collect()


# catboost Yuki
print("load ml_xgb oof ...")
ml_xgb_train = pd.read_parquet('../input/ml_xgb_5fold_train.parquet', engine='pyarrow')
ml_xgb_train = ml_xgb_train[:nrows_train]
ml_xgb_test = pd.read_parquet('../input/ml_xgb_5fold_test.parquet', engine='pyarrow')
ml_xgb_test = ml_xgb_test[:nrows_test]
ml_xgb_full = pd.concat([ml_xgb_train, ml_xgb_test], sort=False)
ml_xgb_full.rename(columns={'ml_xgb_5fold': 'ml_xgb_1'}, inplace=True)
full_df["ml_xgb_1"] = np.array(ml_xgb_full["ml_xgb_1"])
full_df["ml_xgb_1"].clip(0.0, 1.0, inplace=True)
del ml_xgb_train, ml_xgb_test
gc.collect()


# lgbm Yuki
print("load lgbm stacking oof ...")
yuki_oof_stacking_1_train = pd.read_parquet('../input/oof_stacking_level1_lgbm_1_18474_train.parquet', engine='pyarrow')
yuki_oof_stacking_1_train = yuki_oof_stacking_1_train[:nrows_train]
yuki_oof_stacking_1_test = pd.read_parquet('../input/oof_stacking_level1_lgbm_1_18474_test.parquet', engine='pyarrow')
yuki_oof_stacking_1_test = yuki_oof_stacking_1_test[:nrows_test]
yuki_oof_stacking_1_full = pd.concat([yuki_oof_stacking_1_train, yuki_oof_stacking_1_test], sort=False)
yuki_oof_stacking_1_full.rename(columns={'oof_stacking_level1_lgbm_1': 'yuki_oof_stacking_1'}, inplace=True)
full_df["yuki_oof_stacking_1"] = np.array(yuki_oof_stacking_1_full["yuki_oof_stacking_1"])
full_df["yuki_oof_stacking_1"].clip(0.0, 1.0, inplace=True)
del yuki_oof_stacking_1_train, yuki_oof_stacking_1_test
gc.collect()


# lgbm Yuki
print("load lgbm 2 oof ...")
yuki_oof_lgb_2_train = pd.read_csv('../input/oof_stacking_level1_lgbm_no_oof_xentropy_2_33000_train.csv')
yuki_oof_lgb_2_train = yuki_oof_lgb_2_train[:nrows_train]
yuki_oof_lgb_2_test = pd.read_csv('../input/oof_stacking_level1_lgbm_no_oof_xentropy_2_33000_test.csv')
yuki_oof_lgb_2_test = yuki_oof_lgb_2_test[:nrows_test]
yuki_oof_lgb_2_full = pd.concat([yuki_oof_lgb_2_train, yuki_oof_lgb_2_test], sort=False)
yuki_oof_lgb_2_full.rename(columns={'oof_stacking_level1_lgbm_no_oof_xentropy_2': 'yuki_oof_lgb_2'}, inplace=True)
full_df["yuki_oof_lgb_2"] = np.array(yuki_oof_lgb_2_full["yuki_oof_lgb_2"])
full_df["yuki_oof_lgb_2"].clip(0.0, 1.0, inplace=True)
del yuki_oof_lgb_2_train, yuki_oof_lgb_2_test
gc.collect()

# xgbm Yuki
print("load xgb oof ...")
yuki_oof_xgb_2_train = pd.read_csv('../input/xgb_oof_yuki_train.csv')
yuki_oof_xgb_2_train = yuki_oof_xgb_2_train[:nrows_train]
yuki_oof_xgb_2_test = pd.read_csv('../input/xgb_oof_yuki_test.csv')
yuki_oof_xgb_2_test = yuki_oof_xgb_2_test[:nrows_test]
yuki_oof_xgb_2_full = pd.concat([yuki_oof_xgb_2_train, yuki_oof_xgb_2_test], sort=False)
yuki_oof_xgb_2_full.rename(columns={'deal_probability': 'yuki_oof_xgb'}, inplace=True)
full_df["yuki_oof_xgb"] = np.array(yuki_oof_xgb_2_full["yuki_oof_xgb"])
full_df["yuki_oof_xgb"].clip(0.0, 1.0, inplace=True)
del yuki_oof_xgb_2_train, yuki_oof_xgb_2_test
gc.collect()


# mlp Yuki
print("load mlp 2 oof ...")
yuki_oof_mlp_train = pd.read_parquet('../input/oof_stacking_level1_mlp_train.parquet', engine='pyarrow')
yuki_oof_mlp_train = yuki_oof_mlp_train[:nrows_train]
yuki_oof_mlp_test = pd.read_parquet('../input/oof_stacking_level1_mlp_test.parquet', engine='pyarrow')
yuki_oof_mlp_test = yuki_oof_mlp_test[:nrows_test]
yuki_oof_mlp_full = pd.concat([yuki_oof_mlp_train, yuki_oof_mlp_test], sort=False)
yuki_oof_mlp_full.rename(columns={'oof_stacking_level1_mlp': 'yuki_oof_mlp'}, inplace=True)
full_df["yuki_oof_mlp"] = np.array(yuki_oof_mlp_full["yuki_oof_mlp"])
full_df["yuki_oof_mlp"].clip(0.0, 1.0, inplace=True)
del yuki_oof_mlp_train, yuki_oof_mlp_test
gc.collect()


# mlp no oof Yuki
print("load mlp blend oof ...")
yuki_oof_mlp_no_oof_train = pd.read_parquet('../input/oof_stacking_level1_mlp_nooof_base_root_mean_squared_error_train.parquet', engine='pyarrow')
yuki_oof_mlp_no_oof_train = yuki_oof_mlp_no_oof_train[:nrows_train]
yuki_oof_mlp_no_oof_test = pd.read_parquet('../input/oof_stacking_level1_mlp_nooof_base_root_mean_squared_error_test.parquet', engine='pyarrow')
yuki_oof_mlp_no_oof_test = yuki_oof_mlp_no_oof_test[:nrows_test]
yuki_oof_mlp_no_oof_full = pd.concat([yuki_oof_mlp_no_oof_train, yuki_oof_mlp_no_oof_test], sort=False)
yuki_oof_mlp_no_oof_full.rename(columns={'oof_stacking_level1_mlp_nooof_base_root_mean_squared_error': 'yuki_oof_mlp_no_oof'}, inplace=True)
full_df["yuki_oof_mlp_no_oof"] = np.array(yuki_oof_mlp_no_oof_full["yuki_oof_mlp_no_oof"])
full_df["yuki_oof_mlp_no_oof"].clip(0.0, 1.0, inplace=True)
del yuki_oof_mlp_no_oof_train, yuki_oof_mlp_no_oof_test
gc.collect()


# [2559]  train-rmse:0.207816     valid-rmse:0.210334
# FM huiqing
print("load huiqin_fm oof ...")
huiqin_fm_train = pd.read_csv('../input/ftrl_fmV3_train.csv',nrows=nrows_train)
#huiqin_fm_train.drop("user_id", axis=1, inplace=True)
huiqin_fm_test = pd.read_csv('../input/ftrl_fm_V3_test.csv', nrows=nrows_test)
#huiqin_fm_test.drop("user_id", axis=1, inplace=True)
huiqin_fm_full = pd.concat([huiqin_fm_train, huiqin_fm_test], sort=False)
huiqin_fm_full.rename(columns={'deal_probability': 'huiqin_fm_3'}, inplace=True)
full_df = pd.merge(full_df, huiqin_fm_full, on='item_id', how='left')
full_df["huiqin_fm_3"].clip(0.0, 1.0, inplace=True)
del huiqin_fm_train, huiqin_fm_test
gc.collect()


# FM huiqing
print("load huiqin_fm oof 2 ...")
huiqin_fm_train = pd.read_csv('../input/ftrl_fmV1_train.csv',nrows=nrows_train)
#huiqin_fm_train.drop("user_id", axis=1, inplace=True)
huiqin_fm_test = pd.read_csv('../input/ftrl_fm_v1_test.csv', nrows=nrows_test)
#huiqin_fm_test.drop("user_id", axis=1, inplace=True)
huiqin_fm_full = pd.concat([huiqin_fm_train, huiqin_fm_test], sort=False)
huiqin_fm_full.rename(columns={'deal_probability': 'huiqin_fm_2'}, inplace=True)
full_df = pd.merge(full_df, huiqin_fm_full, on='item_id', how='left')
full_df["huiqin_fm_2"].clip(0.0, 1.0, inplace=True)
del huiqin_fm_train, huiqin_fm_test
gc.collect()


# FM huiqing
print("load keras oof 2 ...")
huiqin_fm_train = pd.read_csv('../input/keras_novecV5_train.csv',nrows=nrows_train)
huiqin_fm_train.drop("user_id", axis=1, inplace=True)
huiqin_fm_test = pd.read_csv('../input/keras_novecV5_test.csv', nrows=nrows_test)
#huiqin_fm_test.drop("user_id", axis=1, inplace=True)
huiqin_fm_full = pd.concat([huiqin_fm_train, huiqin_fm_test], sort=False)
huiqin_fm_full.rename(columns={'deal_probability': 'huiqin_keras_2'}, inplace=True)
full_df = pd.merge(full_df, huiqin_fm_full, on='item_id', how='left')
full_df["huiqin_keras_2"].clip(0.0, 1.0, inplace=True)
del huiqin_fm_train, huiqin_fm_test
gc.collect()

# xgb 2
print("load qf_xgb oof ...")
qf_xgb_train = pd.read_csv('../input/qh_xgb_v2_train.csv',nrows=nrows_train)
qf_xgb_train.drop("user_id", axis=1, inplace=True)
qf_xgb_test = pd.read_csv('../input/qh_xgb_v2_test.csv', nrows=nrows_test)
#qf_xgb_test.drop("user_id", axis=1, inplace=True)
qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
qf_xgb_full.rename(columns={'deal_probability': 'qf_xgb_2'}, inplace=True)
full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
full_df["qf_xgb_2"].clip(0.0, 1.0, inplace=True)
del qf_xgb_train, qf_xgb_test
gc.collect()


#print("load qh lgb_2 oof ...")
#qf_xgb_train = pd.read_csv('../input/qh_lgb_v1_oof.csv',nrows=nrows_train)
#qf_xgb_train.drop("user_id", axis=1, inplace=True)
#qf_xgb_test = pd.read_csv('../input/qh_lgb_v1_sub.csv', nrows=nrows_test)
##qf_xgb_test.drop("user_id", axis=1, inplace=True)
#qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
#qf_xgb_full.rename(columns={'deal_probability': 'qh_lgb_s1'}, inplace=True)
#full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
#full_df["qh_lgb_s1"].clip(0.0, 1.0, inplace=True)
#del qf_xgb_train, qf_xgb_test
#gc.collect()


print("load qh lgb_ oof ...")
qf_xgb_train = pd.read_csv('../input/qh_lgb_v5_train.csv',nrows=nrows_train)
qf_xgb_train.drop("user_id", axis=1, inplace=True)
qf_xgb_test = pd.read_csv('../input/qh_lgb_v5_test.csv', nrows=nrows_test)
#qf_xgb_test.drop("user_id", axis=1, inplace=True)
qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
qf_xgb_full.rename(columns={'deal_probability': 'qh_lgb_s5'}, inplace=True)
full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
full_df["qh_lgb_s5"].clip(0.0, 1.0, inplace=True)
del qf_xgb_train, qf_xgb_test
gc.collect()


print("load qh lgb_3 oof ...")
qf_xgb_train = pd.read_csv('../input/qh_lgb_train_v2_oof.csv',nrows=nrows_train)
qf_xgb_train.drop("user_id", axis=1, inplace=True)
qf_xgb_test = pd.read_csv('../input/qh_lgb_subv2_test.csv', nrows=nrows_test)
#qf_xgb_test.drop("user_id", axis=1, inplace=True)
qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
qf_xgb_full.rename(columns={'deal_probability': 'qh_lgb_s2'}, inplace=True)
full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
full_df["qh_lgb_s2"].clip(0.0, 1.0, inplace=True)
del qf_xgb_train, qf_xgb_test
gc.collect()



# xgb 3
#print("load qf_xgb 3 oof ...")
#qf_xgb_train = pd.read_csv('../input/qh_xgb_v3_train.csv',nrows=nrows_train)
#qf_xgb_train.drop("user_id", axis=1, inplace=True)
#qf_xgb_test = pd.read_csv('../input/qh_xgb_v3_test.csv', nrows=nrows_test)
##qf_xgb_test.drop("user_id", axis=1, inplace=True)
#qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
#qf_xgb_full.rename(columns={'deal_probability': 'qf_xgb_3'}, inplace=True)
#full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
#full_df["qf_xgb_3"].clip(0.0, 1.0, inplace=True)
#del qf_xgb_train, qf_xgb_test
#gc.collect()


# lgb mengfei
print("load lgb_mengfei oof ...")
qf_xgb_train = pd.read_csv('../input/ml_lgb_5fold_train_oof.csv',nrows=nrows_train)
qf_xgb_train.drop("user_id", axis=1, inplace=True)
qf_xgb_test = pd.read_csv('../input/ml_lgb_5fold_test.csv', nrows=nrows_test)
#qf_xgb_test.drop("user_id", axis=1, inplace=True)
qf_xgb_full = pd.concat([qf_xgb_train, qf_xgb_test], sort=False)
qf_xgb_full.rename(columns={'deal_probability': 'mengfei_LGB'}, inplace=True)
full_df = pd.merge(full_df, qf_xgb_full, on='item_id', how='left')
full_df["mengfei_LGB"].clip(0.0, 1.0, inplace=True)
full_df["mengfei_LGB"].clip(0.0, 1.0, inplace=True)
del qf_xgb_train, qf_xgb_test
gc.collect()


# ----------------------------------------------------------------------------------------

# keras_vec huiqing
#[2073]  train-rmse:0.209329     valid-rmse:0.211136
print("load keras_vec oof ...")
keras_vec_train = pd.read_csv('../input/keras_vec_train.csv',nrows=nrows_train)
keras_vec_train.drop("user_id", axis=1, inplace=True)
keras_vec_test = pd.read_csv('../input/keras_vec_test_V7.csv', nrows=nrows_test)
#keras_vec_test.drop("user_id", axis=1, inplace=True)
keras_vec_full = pd.concat([keras_vec_train, keras_vec_test], sort=False)
keras_vec_full.rename(columns={'deal_probability': 'keras_vec_1'}, inplace=True)
full_df = pd.merge(full_df, keras_vec_full, on='item_id', how='left')
full_df["keras_vec_1"].clip(0.0, 1.0, inplace=True)
del keras_vec_train, keras_vec_test
gc.collect()


print("load cat v3 oof ...")
cat_v3_train = pd.read_csv('../input/catboost_submissionV3_train.csv',nrows=nrows_train)
cat_v3_train.drop("user_id", axis=1, inplace=True)
cat_v3_test = pd.read_csv('../input/catboost_submissionV3_test.csv', nrows=nrows_test)
#cat_v3_test.drop("user_id", axis=1, inplace=True)
cat_v3_full = pd.concat([cat_v3_train, cat_v3_test], sort=False)
cat_v3_full.rename(columns={'deal_probability': 'cat_v3'}, inplace=True)
full_df = pd.merge(full_df, cat_v3_full, on='item_id', how='left')
full_df["cat_v3"].clip(0.0, 1.0, inplace=True)
del cat_v3_train, cat_v3_test
gc.collect()

# h2o huiqing
print("load h2o oof ...")
huiqin_h2o_train = pd.read_csv('../input/h2o_submissionV2_train.csv',nrows=nrows_train)
huiqin_h2o_train.drop("user_id", axis=1, inplace=True)
huiqin_h2o_test = pd.read_csv('../input/h2o_submissionV2_test.csv', nrows=nrows_test)
#huiqin_h2o_test.drop("user_id", axis=1, inplace=True)
huiqin_h2o_full = pd.concat([huiqin_h2o_train, huiqin_h2o_test], sort=False)
huiqin_h2o_full.rename(columns={'deal_probability': 'huiqin_h2o_2'}, inplace=True)
full_df = pd.merge(full_df, huiqin_h2o_full, on='item_id', how='left')
full_df["huiqin_h2o_2"].clip(0.0, 1.0, inplace=True)
del huiqin_h2o_train, huiqin_h2o_test
gc.collect()

# h2o huiqing
print("load h2o oof ...")
huiqin_h2o_train = pd.read_csv('../input/h2o_submissionV1_train.csv',nrows=nrows_train)
huiqin_h2o_train.drop("user_id", axis=1, inplace=True)
huiqin_h2o_test = pd.read_csv('../input/h2o_submissionV1_test.csv', nrows=nrows_test)
#huiqin_h2o_test.drop("user_id", axis=1, inplace=True)
huiqin_h2o_full = pd.concat([huiqin_h2o_train, huiqin_h2o_test], sort=False)
huiqin_h2o_full.rename(columns={'deal_probability': 'huiqin_h2o_1'}, inplace=True)
full_df = pd.merge(full_df, huiqin_h2o_full, on='item_id', how='left')
full_df["huiqin_h2o_1"].clip(0.0, 1.0, inplace=True)
del huiqin_h2o_train, huiqin_h2o_test
gc.collect()

# rnn open
print("load rnn open oof ...")
huiqin_h2o_train = pd.read_csv('../input/rnn_opensource_train.csv',nrows=nrows_train)
huiqin_h2o_train.drop("user_id", axis=1, inplace=True)
huiqin_h2o_test = pd.read_csv('../input/rnn_opensource_test.csv', nrows=nrows_test)
#huiqin_h2o_test.drop("user_id", axis=1, inplace=True)
huiqin_h2o_full = pd.concat([huiqin_h2o_train, huiqin_h2o_test], sort=False)
huiqin_h2o_full.rename(columns={'deal_probability': 'huiqin_rnn'}, inplace=True)
full_df = pd.merge(full_df, huiqin_h2o_full, on='item_id', how='left')
full_df["huiqin_rnn"].clip(0.0, 1.0, inplace=True)
del huiqin_h2o_train, huiqin_h2o_test
gc.collect()


# zhuang NN huiqing
print("load zhuang_nn oof ...")
zhuang_nn_train = pd.read_csv('../input/2179_5fold_train_oof.csv',nrows=nrows_train)
zhuang_nn_train.drop("user_id", axis=1, inplace=True)
zhuang_nn_test = pd.read_csv('../input/2179_5fold_test.csv', nrows=nrows_test)
#zhuang_nn_test.drop("user_id", axis=1, inplace=True)
zhuang_nn_full = pd.concat([zhuang_nn_train, zhuang_nn_test], sort=False)
zhuang_nn_full.rename(columns={'deal_probability': 'zhuang_nn_1'}, inplace=True)
full_df = pd.merge(full_df, zhuang_nn_full, on='item_id', how='left')
full_df["zhuang_nn_1"].clip(0.0, 1.0, inplace=True)
del zhuang_nn_train, zhuang_nn_test
gc.collect()


# zhuang NN huiqing
print("load zhuang_nn v6 oof ...")
zhuang_nn_train = pd.read_csv('../input/2179_5fold_train_oof_v6.csv',nrows=nrows_train)
zhuang_nn_train.drop("user_id", axis=1, inplace=True)
zhuang_nn_test = pd.read_csv('../input/2179_5fold_test_v6.csv', nrows=nrows_test)
#zhuang_nn_test.drop("user_id", axis=1, inplace=True)
zhuang_nn_full = pd.concat([zhuang_nn_train, zhuang_nn_test], sort=False)
zhuang_nn_full.rename(columns={'deal_probability': 'zhuang_nn_6'}, inplace=True)
full_df = pd.merge(full_df, zhuang_nn_full, on='item_id', how='left')
full_df["zhuang_nn_6"].clip(0.0, 1.0, inplace=True)
del zhuang_nn_train, zhuang_nn_test
gc.collect()


# zhuang NN huiqing
print("load zhuang_nn v6 oof ...")
zhuang_nn_train = pd.read_csv('../input/2179_5fold_train_oof_v5.csv',nrows=nrows_train)
zhuang_nn_train.drop("user_id", axis=1, inplace=True)
zhuang_nn_test = pd.read_csv('../input/2179_5fold_test_v5.csv', nrows=nrows_test)
#zhuang_nn_test.drop("user_id", axis=1, inplace=True)
zhuang_nn_full = pd.concat([zhuang_nn_train, zhuang_nn_test], sort=False)
zhuang_nn_full.rename(columns={'deal_probability': 'zhuang_nn_5'}, inplace=True)
full_df = pd.merge(full_df, zhuang_nn_full, on='item_id', how='left')
full_df["zhuang_nn_5"].clip(0.0, 1.0, inplace=True)
del zhuang_nn_train, zhuang_nn_test
gc.collect()

# zhuang NN huiqing
print("load zhuang_nn 2 oof ...")
zhuang_nn_train = pd.read_csv('../input/v2_2179_5fold_train_oof.csv',nrows=nrows_train)
zhuang_nn_train.drop("user_id", axis=1, inplace=True)
zhuang_nn_test = pd.read_csv('../input/v2_2179_5fold_test.csv', nrows=nrows_test)
#zhuang_nn_test.drop("user_id", axis=1, inplace=True)
zhuang_nn_full = pd.concat([zhuang_nn_train, zhuang_nn_test], sort=False)
zhuang_nn_full.rename(columns={'deal_probability': 'zhuang_nn_2'}, inplace=True)
full_df = pd.merge(full_df, zhuang_nn_full, on='item_id', how='left')
full_df["zhuang_nn_2"].clip(0.0, 1.0, inplace=True)
del zhuang_nn_train, zhuang_nn_test
gc.collect()


# zhuang NN huiqing
print("load zhuang_nn 3 oof ...")
zhuang_nn_train = pd.read_csv('../input/2179_5fold_train_oof_v2.csv',nrows=nrows_train)
zhuang_nn_train.drop("user_id", axis=1, inplace=True)
zhuang_nn_test = pd.read_csv('../input/2179_5fold_test_v2.csv', nrows=nrows_test)
#zhuang_nn_test.drop("user_id", axis=1, inplace=True)
zhuang_nn_full = pd.concat([zhuang_nn_train, zhuang_nn_test], sort=False)
zhuang_nn_full.rename(columns={'deal_probability': 'zhuang_nn_3'}, inplace=True)
full_df = pd.merge(full_df, zhuang_nn_full, on='item_id', how='left')
full_df["zhuang_nn_3"].clip(0.0, 1.0, inplace=True)
del zhuang_nn_train, zhuang_nn_test
gc.collect()


from sklearn.cross_validation import KFold
NFOLDS = 10#5
SEED = 42
kf = KFold(len_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)

list_oof = [x for x in list(full_df) if x not in ["region", 
            "city", 
            "parent_category_name",
            "category_name", 
            "item_seq_number",
            "image_top_1", 
            "price",
            "price+",
            "user_type",
            "item_seq_number+",
            "param_1", "param_2", 
            "param_3",
            "income", "population",
            "region_city",
            ]]

tmp_df = full_df[list_oof]
full_df["max"] = tmp_df.max(axis=1)
full_df["max_median"] = tmp_df.max(axis=1) - tmp_df.median(axis=1)
#full_df["median"] = tmp_df.median(axis=1)
#full_df["min"] = tmp_df.min(axis=1)

#
# Ridge
#from sklearn.linear_model import Ridge
#ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
#                'max_iter':None, 'tol':1e-3, 'solver':'auto', 'random_state':SEED}
#ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
#full_df = pd.DataFrame(full_df)
#tmp2 = pd.DataFrame(full_df)
#print('ridge 1 oof ...')
#ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(full_df)[:len_train], y, np.array(full_df)[len_train:])
#ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
#tmp2['ridge_preds_1'] = ridge_preds.astype(np.float32)
#tmp2['ridge_preds_1'].clip(0.0, 1.0, inplace=True)
#
##full_df['sgd_preds_1'] = tmp1['sgd_preds_1'].astype(np.float32)
#full_df['ridge_preds_1'] = tmp2['ridge_preds_1'].astype(np.float32)


full_df.drop("item_id", axis=1, inplace=True)

full_df.drop(["max","max_median","income", "population", "key_points"], axis=1, inplace=True)

          
train_df = full_df[:len_train]
test_df = full_df[len_train:]

params = {'eta': 0.008,  # 0.03,
          "booster": "gbtree",
          "tree_method": "gpu_hist",
          "gpu_id":0,
          'max_depth': 5,  # 18
          "max_bin": 512,
#          "nthread": 3,
          'subsample': 0.6,
          'colsample_bytree': 0.9,
          'colsample_bylevel': 0.8,
          'min_child_weight': 10,
          'alpha': 1,
          'gamma': 1,
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'random_state': 42,
          'silent': True,   
          # "nrounds": 8000
          }

numIter = 0
rmse_sum = 0
folds=5
from sklearn.model_selection import KFold
kf = KFold(n_splits=folds, random_state=42, shuffle=True)




# SVR
#numIter = 0
#print("doing svr oof ...")
#from sklearn.svm import SVR
#svr = SVR( max_iter=100)
#val_predict = np.zeros(y.shape)
#pred_test_sum = 0
#for train_index, valid_index in kf.split(y):      
#      numIter += 1      
#      print("training in fold " + str(numIter))
#      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
#      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]     
#      svr.fit(X_train, y_train)      
#      pred_val = svr.predict(X_valid)
#      pred_test = svr.predict(test_df)     
#      val_predict[valid_index] = pred_val     
#      print('rmse of svr fold'+ str(numIter) + 'is: ', rmse(y_valid, pred_val))
#      pred_test_sum += pred_test
#pred_test_sum /= folds
#train_df["svr_oof_2"] = val_predict
#test_df["svr_oof_2"] = pred_test_sum



# =============================================================================
# layer 2 KNN
# =============================================================================
# knn 1
#numIter = 0
#print("doing knn oof ...")
#from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=15)
#val_predict = np.zeros(y.shape)
#pred_test_sum = 0
#for train_index, valid_index in kf.split(y):      
#      numIter += 1      
#      print("training in fold " + str(numIter))
#      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
#      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]     
#      neigh.fit(X_train, y_train)      
#      pred_val = neigh.predict(X_valid)
#      pred_test = neigh.predict(test_df)     
#      val_predict[valid_index] = pred_val     
#      print('rmse of knn fold'+ str(numIter) + 'is: ', rmse(y_valid, pred_val))
#      pred_test_sum += pred_test
#pred_test_sum /= folds
#train_df["knn_oof_1"] = val_predict
#test_df["knn_oof_1"] = pred_test_sum
##
### =============================================================================
### layer 3 KNN
### =============================================================================
### knn 2
#numIter = 0
#print("doing knn oof ...")
#from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=10)
#val_predict = np.zeros(y.shape)
#pred_test_sum = 0
#for train_index, valid_index in kf.split(y):      
#      numIter += 1      
#      print("training in fold " + str(numIter))
#      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
#      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]     
#      neigh.fit(X_train, y_train)      
#      pred_val = neigh.predict(X_valid)
#      pred_test = neigh.predict(test_df)     
#      val_predict[valid_index] = pred_val     
#      print('rmse of knn fold'+ str(numIter) + 'is: ', rmse(y_valid, pred_val))
#      pred_test_sum += pred_test
#pred_test_sum /= folds
#train_df["knn_oof_2"] = val_predict
#test_df["knn_oof_2"] = pred_test_sum

### =============================================================================
### layer 3 ridge
### =============================================================================
# ridge 2
from sklearn.linear_model import Ridge
numIter = 0
sum_rmse = 0
print("doing ridge oof ...")
ridge = Ridge(alpha=20.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.0001, solver='auto', random_state=42)
val_predict = np.zeros(y.shape)
pred_test_sum = 0
for train_index, valid_index in kf.split(y):      
      numIter += 1      
      print("training in fold " + str(numIter))
      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]     
      ridge.fit(X_train, y_train)      
      pred_val = ridge.predict(X_valid)
      pred_test = ridge.predict(test_df)     
      val_predict[valid_index] = pred_val     
      print('rmse of ridge fold'+ str(numIter) + 'is: ', rmse(y_valid, pred_val))
      sum_rmse += rmse(y_valid, pred_val)
      pred_test_sum += pred_test
pred_test_sum /= folds
sum_rmse /= folds
train_df["ridge_oof_2"] = val_predict
test_df["ridge_oof_2"] = pred_test_sum


sub_ridge = pd.DataFrame(np.array(test_df["ridge_oof_2"]),columns=["deal_probability"],index=id_test)
sub_ridge["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
sub_ridge.to_csv("stacked_ridge.csv".format(numIter),index=True,header=True)

print("mean rmse is: ", sum_rmse)



### =============================================================================
### layer BayesianRidge
### =============================================================================
# ridge 2
from sklearn import linear_model
numIter = 0
sum_rmse = 0
print("doing B_ridge oof ...")
B_ridge = linear_model.BayesianRidge(n_iter=100)
val_predict = np.zeros(y.shape)
pred_test_sum = 0
for train_index, valid_index in kf.split(y):      
      numIter += 1      
      print("training in fold " + str(numIter))
      X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
      y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]     
      B_ridge.fit(X_train, y_train)      
      pred_val = B_ridge.predict(X_valid)
      pred_test = B_ridge.predict(test_df)     
      val_predict[valid_index] = pred_val     
      print('rmse of ard fold'+ str(numIter) + 'is: ', rmse(y_valid, pred_val))
      sum_rmse += rmse(y_valid, pred_val)
      pred_test_sum += pred_test
pred_test_sum /= folds
sum_rmse /= folds
train_df["b_B_ridge_1"] = val_predict
test_df["b_B_ridge_1"] = pred_test_sum


sub_B_ridge = pd.DataFrame(np.array(test_df["b_B_ridge_1"]),columns=["deal_probability"],index=id_test)
sub_B_ridge["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
sub_B_ridge.to_csv("stacked_b_B_ridge.csv".format(numIter),index=True,header=True)

print("mean rmse is: ", sum_rmse)



# =============================================================================
# layer 5 XGB
# =============================================================================
if mode=='xgboost':
      numIter = 0
      rmse_sum = 0
      pred_test_sum = 0
      val_predict = np.zeros(y.shape)
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
                                    verbose_eval=200)
           
            print("save model ...")
            joblib.dump(model, "xgb_stack_{}.pkl".format(numIter))
      #      load model
      #      lgb_clf = joblib.load("lgb.pkl")
      
            print("Model Evaluation Stage")
            rmse_val = rmse(y_valid, model.predict(Dvalid,ntree_limit=model.best_ntree_limit))
            print( "RMSE:", rmse_val )
            
            xgbpred = model.predict(Dtest,ntree_limit=model.best_ntree_limit)
            
            Dval = xgb.DMatrix(X_valid)
            pred_val = model.predict(Dval, ntree_limit=model.best_ntree_limit)
            val_predict[valid_index] = pred_val 
                  
            xgbsub = pd.DataFrame(xgbpred,columns=["deal_probability"],index=id_test)
            xgbsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
            xgbsub.to_csv("stacked_xgb_5fold_sub_{}.csv".format(numIter),index=True,header=True)
            
            rmse_sum += rmse_val
            
            pred_test_sum += xgbpred
                        
            del X_train, X_valid, y_train, y_valid, Dtrain, Dvalid
            gc.collect()
            
      pred_test_sum /= folds      
      print("mean rmse is:", rmse_sum/folds)
      train_df["deal_probability"] = np.array(val_predict)
      #test_df["deal_probability"] = np.array(pred_test_sum)
      #
      train_df["item_id"] = id_train
      #test_df["item_id"] = id_test
      #
      train_df[["item_id", "deal_probability"]].to_csv("stacked_xgb_5fold_oof.csv", index=False)
      
      
      
      #test_df[["item_id", "deal_probability"]].to_csv("ml_stack_oof_test.csv", index=False)
      
      #tmp = pd.DataFrame()
      #tmp = pd.concat([train_df, test_df])
      #tmp[["city","xgb_here_2"]].to_csv('xgb_final_oof.csv',index=False)
      # =============================================================================
      # feature importance
      # =============================================================================
      def ceate_feature_map(features):
          outfile = open('xgb.fmap', 'w')
          for i, feat in enumerate(features):
              outfile.write('{0}\t{1}\tq\n'.format(i, feat))
          outfile.close()
      
      # XGB feature importances
      # Based on https://www.kaggle.com/mmueller/liberty-mutual-group-property-inspection-prediction/xgb-feature-importance-python/code
      import operator
      import matplotlib.pyplot as plt
          
      features = list(train_df)
      
      ceate_feature_map(features)
      importance = model.get_fscore(fmap='xgb.fmap')
      importance = sorted(importance.items(), key=operator.itemgetter(1))
      
      df = pd.DataFrame(importance, columns=['feature', 'fscore'])
      df['fscore'] = df['fscore'] / df['fscore'].sum()
      
      featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
      plt.title('XGBoost Feature Importance')
      plt.xlabel('relative importance')
      fig_featp = featp.get_figure()
      fig_featp.savefig('feature_importance_xgb.png',bbox_inches='tight',pad_inches=1)


elif mode=='catboost':
      print("Train CatBoost Decision Tree")
      nfold=5
      kf = KFold(n_splits=nfold, random_state=42, shuffle=True)
      val_predict= np.zeros(y.shape)
      aver_rmse=0.0
      numIter = 0
      for train_index, val_index in kf.split(y):
          numIter +=1
          print("training in fold " + str(numIter))
          x_train, x_val = train_df.iloc[train_index], train_df.iloc[val_index]
          y_train, y_val = y.iloc[train_index], y.iloc[val_index]
          model =  cb_model = CatBoostRegressor(iterations=1500,
                                   learning_rate=0.01,
                                   depth=7,
                                   #loss_function='RMSE',
                                   eval_metric='RMSE',
                                   random_seed = 42, # reminder of my mortality
                                   od_type='Iter',
                                   metric_period = 50,
                                   od_wait=50)#20
          model.fit(x_train, y_train,
                       eval_set=(x_val, y_val),
                       #cat_features=categorical_features_pos,
                       use_best_model=True,
                       verbose=True)
          print("start to predict x_train")
          train_pred = model.predict(x_train)
          y_pred = model.predict(x_val)
          val_predict[val_index] = y_pred
          rmses = rmse(y_val, y_pred)
          aver_rmse += rmses
          print('valid score: {}'.format(rmse))

          pred = model.predict(test_df)
         
          catsub = pd.DataFrame(pred,columns=["deal_probability"],index=id_test)
          catsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
          catsub.to_csv("stacked_cat_5fold_sub_{}.csv".format(numIter),index=True,header=True)
           
      print("average rmse:{}".format(aver_rmse/nfold))
      label = ['deal_probability']      
      val_predicts = pd.DataFrame(data=val_predict, columns= label)
      val_predicts['item_id'] = id_train
      val_predicts.to_csv('stacked_cat_5fold_oof.csv', index=False)
      

elif mode=='lightgbm':
      lgbm_params = {
            "tree_learner": "feature",
            "num_threads": 3,
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "regression",  # regression,xentropy
            # "max_delta_step":2,
            "metric": "rmse",
            "max_depth": 10,
            "num_leaves": 128,  # 500, 280,360,500,32
            "feature_fraction": 0.7,  # 0.4
            "bagging_fraction": 0.7,  # 0.4
            "learning_rate": 0.005,  # 0.015
            "verbose": -1,
            'lambda_l1': 1,
            'lambda_l2': 1,
            "max_bin": 32,
            }
      
      rmse_sum = 0
      numIter = 0
      pred_test_sum = 0
      val_predict = np.zeros(y.shape)
      
      cat_col = [
                 "region", 
                 "parent_category_name",
                 "category_name", 
                 "user_type", 
                 "image_top_1",
                 "param_1", 
                 "param_2", 
                 "param_3",
                 "price+",
                 "item_seq_number+",
                 ]
      
      for train_index, valid_index in kf.split(y):      
            numIter += 1      
            print("training in fold " + str(numIter))
            X_train, X_valid = train_df.iloc[train_index], train_df.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            lgtrain = lgb.Dataset(X_train, y_train,
      #                            categorical_feature = cat_col
                                  )
            lgvalid = lgb.Dataset(X_valid, y_valid,
      #                            categorical_feature = cat_col
                                  )
            
            lgb_clf = lgb.train(
                          lgbm_params,
                          lgtrain,
                          num_boost_round=32000,
                          valid_sets=[lgtrain, lgvalid],
                          valid_names=["train","valid"],
                          early_stopping_rounds=200,
                          verbose_eval=100, #200
                          )
           
            print("save model ...")
            joblib.dump(lgb_clf, "lgb_stack_{}.pkl".format(numIter))
      #      load model
      #      lgb_clf = joblib.load("lgb.pkl")
      
            print("Model Evaluation Stage")
            print( "RMSE:", rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)) )
            
            lgpred = lgb_clf.predict(test_df, num_iteration=lgb_clf.best_iteration)
                  
            lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=id_test)
            lgsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
            lgsub.to_csv("stacked_lgb_5fold_sub_{}.csv".format(numIter),index=True,header=True)
            
            pred_val = lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)
            val_predict[valid_index] = pred_val 
            
            rmse_sum += rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration))
            
            pred_test_sum +=lgpred
            gc.collect()
      
      pred_test_sum /= folds
      print("mean rmse is:", rmse_sum/folds)
      
      label = ['deal_probability']      
      val_predicts = pd.DataFrame(data=val_predict, columns=label)
      val_predicts['item_id'] = id_train
      val_predicts.to_csv('stacked_lgb_5fold_oof.csv', index=False)

else:
      pass