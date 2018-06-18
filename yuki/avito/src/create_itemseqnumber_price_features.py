import numpy as np
import pandas as pd
import os
import gc
from joblib import Parallel, delayed
from utils import *

train = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["deal_probability", "image"],axis=1)
test = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["image"],axis=1)
n_train = train.shape[0]
n_test = test.shape[0]
df = pd.concat([train, test])
del train, test; gc.collect()
df["dayofweek"] = df.activation_date.dt.weekday

# price bin features
def cluster_price(p):
    if p < 500:
        return 1
    elif 500<=p<1000:
        return 2
    elif 1000<=p<5000:
        return 3
    elif 5000<=p<10000:
        return 4
    elif 10000<=p<50000:
        return 5
    elif 50000<=p<100000:
        return 6
    elif 100000<=p:
        return 7
    else:
        return 8

def bin_price(p, cluster):
    if cluster==1:
        upper = 500
    elif cluster==2:
        upper = 1000
    elif cluster==3:
        upper = 5000
    elif cluster==4:
        upper = 10000
    elif cluster==5:
        upper = 50000
    elif cluster==6:
        upper = 100000
    elif cluster==7:
        upper = 1000000
    else:
        return 0
    divider = upper/20
    bin = int(p / divider) + 1
    if bin%2==0:
        return -1
    else:
        return 1

df["price_cluster"] = Parallel(n_jobs=-1)([delayed(cluster_price)(p) for p in df.price])
df["price_bin"] = Parallel(n_jobs=-1)([delayed(bin_price)(p,c) for p,c in df[["price", "price_cluster"]].values])
features = np.zeros((df.shape[0], 8))
for i, (c, b) in enumerate(df[["price_cluster", "price_bin"]].values):
    features[i,c-1] = 1 * b
df_out = pd.DataFrame(features, columns=["price_bin_feature_{}".format(i+1) for i in range(8)])
to_parquet(df_out.iloc[:n_train,:], "../features/fe_price_bin_feature_train.parquet")
to_parquet(df_out.iloc[n_train:,:], "../features/fe_price_bin_feature_test.parquet")

# item_seq_number (X price)
df_isn = df.item_seq_number.value_counts().reset_index()
df_isn.columns = ["item_seq_number", "item_seq_number_count"]
df_isn = pd.merge(df_isn, df.groupby("item_seq_number").price.median().\
                            reset_index().rename(columns={"price":"item_seq_number_price_median"})
                            ,on="item_seq_number",how="left")
df_isn = pd.merge(df_isn, df.groupby("item_seq_number").price.mean().\
                            reset_index().rename(columns={"price":"item_seq_number_price_mean"})
                            ,on="item_seq_number",how="left")
df_isn = pd.merge(df_isn, df.groupby("item_seq_number").price.std().\
                            reset_index().rename(columns={"price":"item_seq_number_price_std"})
                            ,on="item_seq_number",how="left")
df_isn = pd.merge(df_isn, df.groupby("item_seq_number").price.max().\
                            reset_index().rename(columns={"price":"item_seq_number_price_max"})
                            ,on="item_seq_number",how="left")
df_isn = pd.merge(df_isn, df.groupby("item_seq_number").price.min().\
                            reset_index().rename(columns={"price":"item_seq_number_price_min"})
                            ,on="item_seq_number",how="left")
df_out = pd.merge(df[["item_seq_number"]], df_isn, on="item_seq_number", how="left").drop("item_seq_number",axis=1)
to_parquet(df_out.iloc[:n_train,:], "../features/fe_itemseqnumber_X_price_train.parquet")
to_parquet(df_out.iloc[n_train:,:], "../features/fe_itemseqnumber_X_price_test.parquet")
