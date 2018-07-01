# -*- coding: utf-8 -*-
"""
ML
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.preprocessing import LabelEncoder


def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def find_nearest(value, array):
    # array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

lbl = LabelEncoder()
cat_col = ["param_1", "param_2", "param_3","user_type"]
def feature_Eng_label_Enc(df):
    print('feature engineering -> label encoding ...')
    for col in cat_col:
        df[col] = lbl.fit_transform(df[col].astype(str))
    gc.collect()
    return df

debug = False

if debug == True:
      df = pd.read_csv("../../input/ml_blend_v7.csv", nrows=3000)
      df.rename(columns ={"deal_probability":"deal_prob_pred"}, inplace=True)
      df_test = pd.read_csv("../../input/test.csv", nrows=3000)
      probs_copy = pd.read_csv("../../input/train.csv", nrows=3000)      
else:
      df = pd.read_csv("../../input/ml_blend_v7.csv")
      df.rename(columns ={"deal_probability":"deal_prob_pred"}, inplace=True)
      df_test = pd.read_csv("../../input/test.csv")
      probs_copy = pd.read_csv("../../input/train.csv")      

feature_Eng_label_Enc(df_test)
feature_Eng_label_Enc(probs_copy)
try:
      df_test['param_hash'] = np.int32(df_test['price']/500) * df_test['param_2'] 
except:
      df_test['param_hash'] = np.NaN
try:
      probs_copy['param_hash'] = np.int32(probs_copy['price']/500) * probs_copy['param_2']
except:
      probs_copy['param_hash'] = np.NaN      
      

param2_list=set(list(df_test.param_hash))

df = df.merge(df_test,on="item_id", how="left")

df["deal_prob_pred"] = df["deal_prob_pred"].astype(np.float32)

tmp_df = pd.DataFrame() 
for col in tqdm(param2_list):          
      probs = probs_copy[probs_copy["param_hash"]==col]
      probs = np.array(probs["deal_probability"])
      df_tmp = df[df["param_hash"]==col]
      df_tmp = df_tmp[["item_id", "deal_prob_pred"]]
      try:
            tmp_array = np.array(df_tmp["deal_prob_pred"].apply(lambda x: find_nearest(x, probs)))   
            df_tmp["deal_prob_processed"] = tmp_array
      except:
            tmp_array = np.array(df_tmp["deal_prob_pred"])
            df_tmp["deal_prob_processed"] = tmp_array
      del tmp_array; gc.collect()
      tmp_df = tmp_df.append(df_tmp)


tmp_df = tmp_df[["item_id", "deal_prob_processed"]]
df = df.merge(tmp_df, on="item_id", how="left")

df.loc[df.deal_prob_processed.isna(), ['deal_prob_processed']] = df.loc[df.deal_prob_processed.isna(), ['deal_prob_pred']].values

df["deal_prob_final"] = (0.4*df["deal_prob_processed"] + 0.6*df["deal_prob_pred"])

sub = pd.DataFrame()
sub["item_id"] = df["item_id"]
sub["deal_probability"] = df["deal_prob_final"]
sub.to_csv("ml_postprocessed_sub.csv", index=False)