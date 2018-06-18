import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import argparse
from scipy.special import erfinv
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from utils import *


X, X_test, _ = read_train_test_data()
# drop label encoding
lbl_cols = [col for col in X.columns if "_labelencod" in col]
X.drop(lbl_cols, axis=1, inplace=True)
X_test.drop(lbl_cols, axis=1, inplace=True)

# nan index
train_nan_idx = csr_matrix((np.isnan(X)).astype(int))
test_nan_idx = csr_matrix((np.isnan(X_test)).astype(int))
X = X.fillna(X.median())#X.fillna(X.median()) # X.fillna(0)
X = X.replace(np.inf, 99999.999)
X = X.replace(-np.inf, -99999.999)
X = X.values
X_test = X_test.fillna(X_test.median())#X_test.fillna(X_test.median())
X_test = X_test.replace(np.inf, 9999.999)
X_test = X_test.replace(-np.inf, -9999.999)
X_test = X_test.values
train_size = X.shape[0]


print("scale data")
scaler = StandardScaler()#StandardScaler() # GaussRankScaler()
X_all = scaler.fit_transform(np.r_[X, X_test])
del X, X_test; gc.collect()
# X = pd.DataFrame(X_all[:train_size,:])
X = pd.DataFrame(X_all[:train_size,:] * np.array((train_nan_idx.todense()==0).astype(int)))
del train_nan_idx
print("Done scaling train data...")
# X_test = pd.DataFrame(X_all[train_size:,:])
X_test = pd.DataFrame(X_all[train_size:,:] * np.array((test_nan_idx.todense()==0).astype(int)))
print("Done scaling test data...")
# del X_all; gc.collect()
del X_all, test_nan_idx;gc.collect()


X_test = X_test.values
X = X.values
y = pd.read_csv("../input/train.csv")["deal_probability"].values

oof_sgd(X, X_test, y, "stacking_elasticnet")

### no oof features
X, X_test, _ = read_train_test_data_all()
# drop label encoding
lbl_cols = [col for col in X.columns if "_labelencod" in col or "oof_" in col]
X.drop(lbl_cols, axis=1, inplace=True)
X_test.drop(lbl_cols, axis=1, inplace=True)

# nan index
train_nan_idx = csr_matrix((np.isnan(X)).astype(int))
test_nan_idx = csr_matrix((np.isnan(X_test)).astype(int))
X = X.fillna(X.median())#X.fillna(X.median()) # X.fillna(0)
X = X.replace(np.inf, 99999.999)
X = X.replace(-np.inf, -99999.999)
X = X.values
X_test = X_test.fillna(X_test.median())#X_test.fillna(X_test.median())
X_test = X_test.replace(np.inf, 9999.999)
X_test = X_test.replace(-np.inf, -9999.999)
X_test = X_test.values
train_size = X.shape[0]


print("scale data")
scaler = StandardScaler()#StandardScaler() # GaussRankScaler()
X_all = scaler.fit_transform(np.r_[X, X_test])
del X, X_test; gc.collect()
# X = pd.DataFrame(X_all[:train_size,:])
X = pd.DataFrame(X_all[:train_size,:] * np.array((train_nan_idx.todense()==0).astype(int)))
del train_nan_idx
print("Done scaling train data...")
# X_test = pd.DataFrame(X_all[train_size:,:])
X_test = pd.DataFrame(X_all[train_size:,:] * np.array((test_nan_idx.todense()==0).astype(int)))
print("Done scaling test data...")
# del X_all; gc.collect()
del X_all, test_nan_idx;gc.collect()


X_test = X_test.values
X = X.values
y = pd.read_csv("../input/train.csv")["deal_probability"].values

oof_sgd(X, X_test, y, "stacking_elasticnet_nooof")
