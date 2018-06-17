# Catboost for Avito Demand Prediction Challenge
# https://www.kaggle.com/c/avito-demand-prediction
# By Nick Brooks, April 2018
#https://www.kaggle.com/nicapotato/simple-catboost/code
import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
from sklearn.model_selection import KFold
# print("Data:\n", os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import *
# Viz
import seaborn as sns
import matplotlib.pyplot as plt

print("\nData Load Stage")
training = pd.read_csv('../input/train.csv', index_col="item_id", parse_dates=["activation_date"])
traindex = training.index
len_train = len(training)
testing = pd.read_csv('../input/test.csv', index_col="item_id", parse_dates=["activation_date"])
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("\nCreate Time Variables")
df["Year"] = df["activation_date"].dt.year
df["Date of Year"] = df['activation_date'].dt.dayofyear  # Day of Year
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day
df["Quarter"] = df['activation_date'].dt.quarter

# Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type",
               "image_top_1"]
messy_categorical = ["param_1", "param_2", "param_3",
                     # "title", "description"
                     ]  # Need to find better technique for these
print("Encoding :", categorical + messy_categorical)


def char_analyzer(text):
    """
    This is used to split strings in small lots
    anttip saw this in an article
    so <talk> and <talking> would have <Tal> <alk> in common
    should be similar to russian I guess
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
df['desc']=df['title']+df['description']
feature_cnt = 20

tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
train_tfidf = pd.DataFrame(tfidf.fit_transform(df['desc'].astype(str)).toarray())

cols = train_tfidf.columns
for i in range(feature_cnt):
    df['desc' + '_tfidf_' + str(i)] = train_tfidf[cols[i]]

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

## ADD
# 1. Lower Byte encoding
# 2. Russian TD-IDF/ Count Vectorizer
ex_col = ['item_id','user_id','deal_probability','description','title','desc','mean_y']
col = [c for c in df.columns if c not in ex_col]

print("\nCatboost Modeling Stage")

X = df[col].loc[traindex, :].copy()
# X =np.array(df[col][:len_train])
print("x",X.shape)
print("x columns",X.columns)
print("Training Set shape", X.shape)

test = df[col].loc[testdex, :].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()


# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


categorical_features_pos = column_index(X, categorical + messy_categorical)

# Train Model
print("Train CatBoost Decision Tree")
modelstart = time.time()
nfold=5
kf = KFold(n_splits=nfold, random_state=42, shuffle=True)
val_predict= np.zeros(y.shape)
aver_rmse=0.0
fold_id = -1
X =np.array(X)
for train_index, val_index in kf.split(X):
    print(len(train_index),len(val_index))
    fold_id += 1
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    model =  CatBoostRegressor(iterations=700,
                             learning_rate=0.01,
                             depth=16,
                             eval_metric='RMSE',
                             random_seed = 23,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
    model.fit(x_train, y_train,
                 eval_set=(x_val, y_val),
                 cat_features=categorical_features_pos,
                 use_best_model=True,
                 verbose=True)
    print("start to predict x_train")
    train_pred = model.predict(x_train)
    y_pred = model.predict(x_val)
    val_predict[val_index] = y_pred
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    aver_rmse+=rmse
    print('valid score: {}'.format(rmse))
    sub = pd.read_csv('../input/sample_submission.csv')#, nrows=10000*5
    pred = model.predict(test)
    sub['deal_probability'] = pred
    sub['deal_probability'].clip(0.0, 1.0, inplace=True)
    print("Output Prediction CSV")
    sub.to_csv('subm/catboost_submissionV1_{}.csv'.format(fold_id), index=False)

# Feature Importance
# feat_imp = cb_model.get_feature_importance(X=X_train, y=y_train,
#                                            cat_features=categorical_features_pos, fstr_type='FeatureImportance')
# f, ax = plt.subplots(figsize=[8, 4])
# sns.barplot(y=X.columns, x=feat_imp, ax=ax)
# ax.set_title("Feature Importance")
# ax.set_xlabel("Importance")
# plt.savefig('feature_import.png')

print("average rmse:{}".format(aver_rmse/nfold))
train_data = pd.read_csv('../input/train.csv')
label = ['deal_probability']
train_user_ids = train_data.user_id.values
train_item_ids = train_data.item_id.values

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_item_ids.reshape(len(train_user_ids), 1)
val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv('subm/catboost_submissionV1_train.csv', index=False)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))
'''
1w
average rmse:0.24040608183097817
Notebook Runtime: 5.79 Minutes
'''