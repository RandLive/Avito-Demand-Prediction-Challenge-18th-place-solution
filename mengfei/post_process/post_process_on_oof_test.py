# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(ML)s
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import gc


def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def find_nearest(p, probs):
    k = 1000
    sol = 0
    for pro in probs:
        tmp = np.abs(p-pro)
        if tmp < k:
            k = tmp
            sol = pro
    return sol

debug = False
if debug == True:
      df = pd.read_csv("../../input/ml_blend_v7.csv", nrows=3000)
      df["oof_stacking_level1_lgbm_no_oof_xentropy_2"] = df["deal_probability"]
      del df["deal_probability"]
      df_test = pd.read_csv("../../input/test.csv", nrows=3000)
      probs_copy = pd.read_csv("../../input/train.csv", nrows=3000)      
#      y =  pd.read_csv("../../input/train.csv", usecols=["deal_probability"], nrows=3000)["deal_probability"]
      
if debug == False:
      df = pd.read_csv("../../input/ml_blend_v7.csv")
      df["oof_stacking_level1_lgbm_no_oof_xentropy_2"] = df["deal_probability"]
      del df["deal_probability"]
      df_test = pd.read_csv("../../input/test.csv")
      probs_copy = pd.read_csv("../../input/train.csv")      
#      y =  pd.read_csv("../../input/train.csv", usecols=["deal_probability"])["deal_probability"]

param2_list=set(list(df_test.param_2))

#df.drop("user_id", axis=1, inplace=True)

df = df.merge(df_test,on="item_id", how="left")

#print("rmse before: ", rmse(df["oof_stacking_level1_lgbm_no_oof_xentropy_2"], y) )



df["oof_stacking_level1_lgbm_no_oof_xentropy_2"] = df["oof_stacking_level1_lgbm_no_oof_xentropy_2"].astype(np.float32)

for col in tqdm(param2_list):
      tmp_df = pd.DataFrame()
            
      probs = probs_copy[probs_copy["param_2"]==col]
      probs = np.array(probs["deal_probability"])
##      print(probs)
      df_tmp = df[df["param_2"]==col]
      df_tmp = df_tmp[["item_id", "oof_stacking_level1_lgbm_no_oof_xentropy_2"]]
#      
      df_tmp["new_dp"] = df_tmp["oof_stacking_level1_lgbm_no_oof_xentropy_2"].apply(lambda x: find_nearest(x, probs))      
      tmp_df = tmp_df.append(df_tmp)


tmp_df = tmp_df[["item_id", "new_dp"]]
df = df.merge(tmp_df, on="item_id", how="left")

df.loc[df.new_dp.isna(), ['new_dp']] = df.loc[df.new_dp.isna(), ['oof_stacking_level1_lgbm_no_oof_xentropy_2']].values

#print("rmse after: ", rmse(df["new_dp"], y) )
df["new_aver_dp"] = (0.085*df["new_dp"] + 0.95*df["oof_stacking_level1_lgbm_no_oof_xentropy_2"])
#print("rmse after aver: ", rmse(df["new_aver_dp"], y) )

sub = pd.DataFrame()
sub["item_id"] = df["item_id"]
sub["deal_probability"] = df["new_aver_dp"]
sub.to_csv("ml_postprocessed_sub.csv", index=False)
