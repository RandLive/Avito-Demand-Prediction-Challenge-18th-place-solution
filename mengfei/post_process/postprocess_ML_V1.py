# -*- coding: utf-8 -*-
"""
ML
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def find_nearest(p, probs):
    k = 1
    sol = 0
    for pro in probs:
        tmp = np.abs(p-pro)
        if tmp < k:
            k = tmp
            sol = pro
    return sol

debug = True
if debug == True:
      df = pd.read_csv("../../input/ml_blend_v7.csv", nrows=3000)
      df.rename(columns ={"deal_probability":"deal_prob_pred"}, inplace=True)
      df_test = pd.read_csv("../../input/test.csv", nrows=3000)
      probs_copy = pd.read_csv("../../input/train.csv", nrows=3000)      

      
if debug == False:
      df = pd.read_csv("../../input/ml_blend_v7.csv")
      df.rename(columns ={"deal_probability":"deal_prob_pred"}, inplace=True)
      df_test = pd.read_csv("../../input/test.csv")
      probs_copy = pd.read_csv("../../input/train.csv")      

param2_list=set(list(df_test.param_2))

df = df.merge(df_test,on="item_id", how="left")

df["deal_prob_pred"] = df["deal_prob_pred"].astype(np.float32)

for col in tqdm(param2_list):
      tmp_df = pd.DataFrame()            
      probs = probs_copy[probs_copy["param_2"]==col]
      probs = np.array(probs["deal_probability"])
      df_tmp = df[df["param_2"]==col]
      df_tmp = df_tmp[["item_id", "deal_prob_pred"]]
      df_tmp["deal_prob_processed"] = df_tmp["deal_prob_pred"].apply(lambda x: find_nearest(x, probs))      
      tmp_df = tmp_df.append(df_tmp)


tmp_df = tmp_df[["item_id", "deal_prob_processed"]]
df = df.merge(tmp_df, on="item_id", how="left")

df.loc[df.deal_prob_processed.isna(), ['deal_prob_processed']] = df.loc[df.deal_prob_processed.isna(), ['deal_prob_pred']].values
df["deal_prob_final"] = (0.085*df["deal_prob_processed"] + 0.95*df["deal_prob_pred"])

sub = pd.DataFrame()
sub["item_id"] = df["item_id"]
sub["deal_probability"] = df["deal_prob_final"]
sub.to_csv("ml_postprocessed_sub.csv", index=False)