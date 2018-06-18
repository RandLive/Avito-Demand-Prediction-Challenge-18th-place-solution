import os, sys
import gc
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import dill
import requests
import lightgbm as lgb
import traceback
from joblib import Parallel, delayed
import re
from utils import *

X_train, X_test, _ = read_train_test_data()
X = pd.concat([X_train, X_test])
del X_train, X_test; gc.collect()

train_columns = list(X.columns)
X_nonnan = X.dropna(axis=1)
X_nonnan = X_nonnan.replace(np.inf, 9999.999)
X_nonnan = X_nonnan.replace(-np.inf, -9999.999)
not_nan_columns = list(X_nonnan.columns)
nan_columns = [col for col in train_columns if col not in not_nan_columns]
print("Not Nan Columns: {} columns.".format(len(not_nan_columns)))
print(not_nan_columns)
print("Nan Columns: {} columns.".format(len(nan_columns)))
print(nan_columns)


print("PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=len(not_nan_columns)-1, whiten=True)
X_nonnan_pca = pca.fit_transform(X_nonnan)
print("Explained Variance Ratio...")
print(pca.explained_variance_ratio_)
print("Explained Variance Ratio Cumulative sum...")
print(np.cumsum(pca.explained_variance_ratio_))

print("Correlation")
X_corr = X_nonnan.corr()
to_parquet(X_corr, "../tmp/X_corr.parquet")
