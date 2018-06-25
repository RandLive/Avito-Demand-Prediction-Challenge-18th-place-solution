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
# import seaborn as sns
# import matplotlib.pyplot as plt

print("\nData Load Stage")
debug=False
if debug:

    nrows=10000*1
else:
    nrows=1503424
training = pd.read_csv('../input/train.csv',nrows=nrows, index_col="item_id", parse_dates=["activation_date"])
traindex = training.index
len_train = len(training)
testing = pd.read_csv('../input/test.csv',nrows=nrows, index_col="item_id", parse_dates=["activation_date"])
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)


import pickle
with open('../input/train_image_features.p', 'rb') as f:
	x = pickle.load(f)

train_blurinesses = x['blurinesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
	x = pickle.load(f)

test_blurinesses = x['blurinesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_blurinesses, columns=['blurinesses'])
incep_test_image_df = pd.DataFrame(test_blurinesses, columns=['blurinesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding whitenesses ...')
with open('../input/train_image_features.p', 'rb') as f:
	x = pickle.load(f)

train_whitenesses = x['whitenesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
	x = pickle.load(f)

test_whitenesses = x['whitenesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_whitenesses, columns=['whitenesses'])
incep_test_image_df = pd.DataFrame(test_whitenesses, columns=['whitenesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding dullnesses ...')
with open('../input/train_image_features.p', 'rb') as f:
	x = pickle.load(f)

train_dullnesses = x['dullnesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
	x = pickle.load(f)

test_dullnesses = x['dullnesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_dullnesses, columns=['dullnesses'])
incep_test_image_df = pd.DataFrame(test_dullnesses, columns=['dullnesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding average_pixel_width ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_average_pixel_width = x['average_pixel_width']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_average_pixel_width = x['average_pixel_width']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_average_pixel_width, columns=['average_pixel_width'])
incep_test_image_df = pd.DataFrame(test_average_pixel_width, columns=['average_pixel_width'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding average_reds ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_average_reds = x['average_reds']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_average_reds = x['average_reds']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_average_reds, columns=['average_reds'])
incep_test_image_df = pd.DataFrame(test_average_reds, columns=['average_reds'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding average_blues ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_average_blues = x['average_blues']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_average_blues = x['average_blues']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_average_blues, columns=['average_blues'])
incep_test_image_df = pd.DataFrame(test_average_blues, columns=['average_blues'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding average_greens ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_average_greens = x['average_greens']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_average_greens = x['average_greens']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_average_greens, columns=['average_greens'])
incep_test_image_df = pd.DataFrame(test_average_greens, columns=['average_greens'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding widths ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_widths = x['widths']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_widths = x['widths']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_widths, columns=['widths'])
incep_test_image_df = pd.DataFrame(test_widths, columns=['widths'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

print('adding heights ...')
with open('../input/train_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

train_heights = x['heights']
train_ids = x['ids']

with open('../input/test_image_features_1.p', 'rb') as f:
	x = pickle.load(f)

test_heights = x['heights']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_heights, columns=['heights'])
incep_test_image_df = pd.DataFrame(test_heights, columns=['heights'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
training = training.join(incep_train_image_df.set_index('image'), on='image')
testing = testing.join(incep_test_image_df.set_index('image'), on='image')

# ==============================================================================
# image features by Qifeng
# ==============================================================================
print('adding image features ...')
with open('../input/train_image_features_cspace.p', 'rb') as f:
	x = pickle.load(f)

x_train = pd.DataFrame(x, columns=['average_HSV_Ss', \
								   'average_HSV_Vs', \
								   'average_LUV_Ls', \
								   'average_LUV_Us', \
								   'average_LUV_Vs', \
								   'average_HLS_Hs', \
								   'average_HLS_Ls', \
								   'average_HLS_Ss', \
								   'average_YUV_Ys', \
								   'average_YUV_Us', \
								   'average_YUV_Vs', \
								   'ids'
								   ])
# x_train.rename(columns = {'$ids':'image'}, inplace = True)
print('average_HSV_Ss 0',x_train['average_HSV_Ss'][0])
with open('../input/test_image_features_cspace.p', 'rb') as f:
	x = pickle.load(f)

x_test = pd.DataFrame(x, columns=['average_HSV_Ss', \
								  'average_HSV_Vs', \
								  'average_LUV_Ls', \
								  'average_LUV_Us', \
								  'average_LUV_Vs', \
								  'average_HLS_Hs', \
								  'average_HLS_Ls', \
								  'average_HLS_Ss', \
								  'average_YUV_Ys', \
								  'average_YUV_Us', \
								  'average_YUV_Vs', \
								  'ids'
								  ])
# x_test.rename(columns = {'$ids':'image'}, inplace = True)

training = training.join(x_train.set_index('ids'), on='image')
testing = testing.join(x_test.set_index('ids'), on='image')
del x, x_train, x_test;
gc.collect()

#==============================================================================
# image features v2 by Qifeng
#==============================================================================
# print('adding image features ...')
# with open('../input/train_image_features_cspace_v2.p','rb') as f:
#     x = pickle.load(f)
#
# x_train = pd.DataFrame(x, columns = ['average_LAB_Ls',\
#                                      'average_LAB_As',\
#                                      'average_LAB_Bs',\
#                                      'average_YCrCb_Ys',\
#                                      'average_YCrCb_Crs',\
#                                      'average_YCrCb_Cbs',\
#                                      'ids'
#                                      ])
# #x_train.rename(columns = {'$ids':'image'}, inplace = True)
#
# with open('../input/test_image_features_cspace_v2.p','rb') as f:
#     x = pickle.load(f)
#
# x_test = pd.DataFrame(x, columns = ['average_LAB_Ls',\
#                                      'average_LAB_As',\
#                                      'average_LAB_Bs',\
#                                      'average_YCrCb_Ys',\
#                                      'average_YCrCb_Crs',\
#                                      'average_YCrCb_Cbs',\
#                                      'ids'
#                                      ])
# #x_test.rename(columns = {'$ids':'image'}, inplace = True)
#
# training = training.join(x_train.set_index('ids'), on='image')
# testing = testing.join(x_test.set_index('ids'), on='image')
# del x, x_train, x_test; gc.collect()

# =============================================================================
# add geo info: https://www.kaggle.com/frankherfert/avito-russian-region-cities/data
# =============================================================================
# tmp = pd.read_csv("../input/avito_region_city_features.csv",nrows=nrows, usecols=["region", "city", "latitude","longitude"])
# training = training.merge(tmp, on=["city","region"], how="left")
# training["lat_long"] = training["latitude"]+training["longitude"]
# testing = testing.merge(tmp, on=["city","region"], how="left")
# testing["lat_long"] = testing["latitude"]+testing["longitude"]
# print('lat_long 0',training["lat_long"][0] )
# del tmp; gc.collect()

# =============================================================================
# Add region-income
# =============================================================================
# tmp = pd.read_csv("../input/city_population_wiki_v3.csv",nrows=nrows,)
# training = training.merge(tmp, on="city", how="left")
# testing = testing.merge(tmp, on="city", how="left")
# del tmp; gc.collect()
#
#
# # =============================================================================
# # Here Based on https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/code
# # =============================================================================
# used_cols = ["item_id", "user_id"]
# train_active = pd.read_csv("../input/train_active.csv",nrows=nrows, usecols=used_cols)
# test_active = pd.read_csv("../input/test_active.csv",nrows=nrows, usecols=used_cols)
# train_periods = pd.read_csv("../input/periods_train.csv",nrows=nrows, parse_dates=["date_from", "date_to"])
# test_periods = pd.read_csv("../input/periods_test.csv",nrows=nrows, parse_dates=["date_from", "date_to"])
# all_samples = pd.concat([training,train_active,testing,test_active]).reset_index(drop=True)
# all_samples.drop_duplicates(["item_id"], inplace=True)
# del train_active, test_active; gc.collect()
#
# all_periods = pd.concat([train_periods,test_periods])
# del train_periods, test_periods; gc.collect()
#
# all_periods["days_up"] = (all_periods["date_to"] - all_periods["date_from"]).dt.days
# gp = all_periods.groupby(["item_id"])[["days_up"]]
#
# gp_df = pd.DataFrame()
# gp_df["days_up_sum"] = gp.sum()["days_up"]
# gp_df["times_put_up"] = gp.count()["days_up"]
# gp_df.reset_index(inplace=True)
# gp_df.rename(index=str, columns={"index": "item_id"})
#
# all_periods.drop_duplicates(["item_id"], inplace=True)
# all_periods = all_periods.merge(gp_df, on="item_id", how="left")
# all_periods = all_periods.merge(all_samples, on="item_id", how="left")
#
# gp = all_periods.groupby(["user_id"])[["days_up_sum", "times_put_up"]].mean().reset_index()\
# .rename(index=str, columns={"days_up_sum": "avg_days_up_user",
#                             "times_put_up": "avg_times_up_user"})
#
# n_user_items = all_samples.groupby(["user_id"])[["item_id"]].count().reset_index() \
# .rename(index=str, columns={"item_id": "n_user_items"})
# gp = gp.merge(n_user_items, on="user_id", how="outer") #left
#
# del all_samples, all_periods, n_user_items
# gc.collect()
#
# training = training.merge(gp, on="user_id", how="left")
# testing = testing.merge(gp, on="user_id", how="left")
#
# agg_cols = list(gp.columns)[1:]
#
# del gp; gc.collect()
#
# for col in agg_cols:
#     training[col].fillna(-1, inplace=True)
#     testing[col].fillna(-1, inplace=True)
#
# print("merging supplimentary data done!")

print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"] + 0.001)
df["price"].fillna(-999, inplace=True)
df["image_top_1"].fillna(-999, inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)
#picture features

#counting
import string
count = lambda l1, l2: sum([1 for x in l1 if x in l2])
def handle_missing_inplace(dataset):
    dataset['description'].fillna(value='na', inplace=True)
    # dataset["image"].fillna("noinformation", inplace=True)
    dataset["param_1"].fillna("nicapotato", inplace=True)
    dataset["param_2"].fillna("nicapotato", inplace=True)
    dataset["param_3"].fillna("nicapotato", inplace=True)

    dataset['image_top_1'].fillna(value=-1, inplace=True)
    dataset['price'].fillna(value=0, inplace=True)

    columns=['average_HSV_Ss', \
     'average_HSV_Vs', \
     'average_LUV_Ls', \
     'average_LUV_Us', \
     'average_LUV_Vs', \
     'average_HLS_Hs', \
     'average_HLS_Ls', \
     'average_HLS_Ss', \
     'average_YUV_Ys', \
     'average_YUV_Us', \
     'average_YUV_Vs', \

     ]
    for c in columns:
        dataset[c].fillna(-1, inplace=True)

handle_missing_inplace(df)

df["num_desc_punct"] = df["description"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
df["num_desc_capE"] = df["description"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
df["num_desc_capP"] = df["description"].apply(lambda x: count(x, "[А-Я]")).astype(np.int16)

df["num_title_punct"] = df["title"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
df["num_title_capE"] = df["title"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
df["num_title_capP"] = df["title"].apply(lambda x: count(x, "[А-Я]")).astype(np.int16)
# good, used, bad ... count
df["is_in_desc_хорошо"] = df["description"].str.contains("хорошо").map({True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_Плохо"] = df["description"].str.contains("Плохо").map({True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_новый"] = df["description"].str.contains("новый").map({True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_старый"] = df["description"].str.contains("старый").map({True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_используемый"] = df["description"].str.contains("используемый").map(
	{True: 1, False: 0}).astype(
	np.uint8)
df["is_in_desc_есплатная_доставка"] = df["description"].str.contains("есплатная доставка").map(
	{True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_есплатный_возврат"] = df["description"].str.contains("есплатный возврат").map(
	{True: 1, False: 0}).astype(np.uint8)
df["is_in_desc_идеально"] = df["description"].str.contains("идеально").map({True: 1, False: 0}).astype(
	np.uint8)
df["is_in_desc_подержанный"] = df["description"].str.contains("подержанный").map({True: 1, False: 0}).astype(
	np.uint8)
df["is_in_desc_пСниженные_цены"] = df["description"].str.contains("Сниженные цены").map(
	{True: 1, False: 0}).astype(np.uint8)
# df["num_desc_Exclamation"] = df["description"].apply(lambda x: count(x, "!")).astype(np.int16)
# df["num_desc_Question"] = df["description"].apply(lambda x: count(x, "?")).astype(np.int16)

#text feature
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def char_analyzer(text):
    """
    This is used to split strings in small lots
    anttip saw this in an article
    so <talk> and <talking> would have <Tal> <alk> in common
    should be similar to russian I guess
    """
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
# char_vectorizer = TfidfVectorizer(
#             sublinear_tf=True,
#             strip_accents='unicode',
#             tokenizer=char_analyzer,
#             analyzer='word',
#             ngram_range=(1, 4),#
#             max_features=50000)
# X = char_vectorizer.fit_transform(df['title'])
# df['title_word_vec']=X
# del (X)
#another features

df['desc_len'] = np.log1p(df['description'].apply(lambda x: len(x)))
df['title_len'] = np.log1p(df['title'].apply(lambda x: len(x)))
# df['title_desc_len_ratio'] = np.log1p(df['title_len'] / df['desc_len'])
# #
df['desc_word_count'] = df['description'].apply(lambda x: len(x.split()))
# df['mean_des'] = df['description'].apply(
#     lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
# df['mean_title'] = df['title'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10


df["price+"] = np.round(df["price"] * 2.8).astype(np.int16)  # 4.8
df["item_seq_number+"] = np.round(df["item_seq_number"] / 100).astype(np.int16)


print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type",
               "image_top_1"]
messy_categorical = ["param_1", "param_2", "param_3", "title", "description"]  # Need to find better technique for these
print("Encoding :", categorical + messy_categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))

## ADD
# 1. Lower Byte encoding
# 2. Russian TD-IDF/ Count Vectorizer

print("\nCatboost Modeling Stage")

X = df.loc[traindex, :].copy()
print("Training Set shape", X.shape)
test = df.loc[testdex, :].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()

# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(X,categorical + messy_categorical)

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
    model =  cb_model = CatBoostRegressor(iterations=900,
                             learning_rate=0.08,
                             depth=10,
                             #loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed = 23, # reminder of my mortality
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=50)#20
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
    sub = pd.read_csv('../input/sample_submission.csv',nrows=nrows)#, nrows=10000*5
    pred = model.predict(test)
    sub['deal_probability'] = pred
    sub['deal_probability'].clip(0.0, 1.0, inplace=True)
    print("Output Prediction CSV")
    sub.to_csv('subm/catboost_submissionV2_{}.csv'.format(fold_id), index=False)

# Feature Importance
# feat_imp = cb_model.get_feature_importance(X=X_train, y=y_train,
#                                            cat_features=categorical_features_pos, fstr_type='FeatureImportance')
# f, ax = plt.subplots(figsize=[8, 4])
# sns.barplot(y=X.columns, x=feat_imp, ax=ax)
# ax.set_title("Feature Importance")
# ax.set_xlabel("Importance")
# plt.savefig('feature_import.png')

print("average rmse:{}".format(aver_rmse/nfold))
train_data = pd.read_csv('../input/train.csv',nrows=nrows)
label = ['deal_probability']
train_user_ids = train_data.user_id.values
train_item_ids = train_data.item_id.values

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_item_ids.reshape(len(train_user_ids), 1)
val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv('subm/catboost_submissionV2_train.csv', index=False)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))
'''
1w
average rmse:0.24040608183097817
Notebook Runtime: 5.79 Minutes

average rmse:0.2378800346823608
Model Runtime: 0.95 Minutes
Notebook Runtime: 1.40 Minutes

average rmse:0.2376487183004941
Model Runtime: 1.43 Minutes
Notebook Runtime: 2.03 Minutes

average rmse:0.23717375234882132
Model Runtime: 1.46 Minutes
Notebook Runtime: 2.08 Minutes

average rmse:0.23685193911053046
Model Runtime: 2.17 Minutes
Notebook Runtime: 2.79 Minutes

average rmse:0.2369477700838154
Model Runtime: 2.83 Minutes
Notebook Runtime: 3.45 Minutes

average rmse:0.23680960250407868
Model Runtime: 2.30 Minutes
Notebook Runtime: 2.92 Minutes

average rmse:0.23660718248953333
Model Runtime: 2.62 Minutes
Notebook Runtime: 3.24 Minutes
'''