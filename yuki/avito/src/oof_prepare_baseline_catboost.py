import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from utils import *
import argparse
from sklearn import preprocessing

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

# specify the version.
parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', default='ensemble', help='version')
args = parser.parse_args()
version = args.version
is_read_data = True

train = pd.read_csv('../input/train.csv', parse_dates=["activation_date"])
test = pd.read_csv('../input/test.csv', parse_dates=["activation_date"])
y = train.deal_probability.values.copy()
train_user_ids = train.user_id.tolist()
train_item_ids = train.item_id.tolist()
test_user_ids = test.user_id.tolist()
test_item_ids = test.item_id.tolist()
n_train = train.shape[0]
df = pd.concat([train, test], axis=0)
gc.collect()

df["price"] = np.log(df["price"] + 0.001)
df["price"].fillna(-999, inplace=True)
df["image_top_1"].fillna(-999, inplace=True)
df["Weekday"] = df['activation_date'].dt.weekday
df.drop(["activation_date", "image"], axis=1, inplace=True)

categorical = ["Weekday","region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type", "image_top_1", "param_1", "param_2", "param_3"]

df = df[["price"]+categorical]
print("Encoding :", categorical )

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].fillna("_NAN_").astype(str))

X_train = df.iloc[:n_train, :]
X_test = df.iloc[n_train:, :]

# user feature,
train_others = train[["user_id", "item_id"]]
test_others = test[["user_id", "item_id"]]
import glob
user_features = glob.glob("../features/*col1_user_id*")
for f in user_features+["../features/fe_user_price_base.parquet"]:
    train_others = pd.merge(train_others, read_parquet(f,index=False), on="user_id", how="left")
    test_others = pd.merge(test_others, read_parquet(f,index=False), on="user_id", how="left")


train_others = pd.merge(train_others, read_parquet("../features/fe_item_price_pred_diff.parquet",index=False), on="item_id", how="left")
test_others = pd.merge(test_others, read_parquet("../features/fe_item_price_pred_diff.parquet",index=False), on="item_id", how="left")

# image features
img_features = glob.glob("../features/*img_4*train.parquet")
for f in sorted(img_features):
    train_others = pd.concat([train_others, read_parquet(f,index=False)], axis=1)
img_features = glob.glob("../features/*img_4*test.parquet")
for f in sorted(img_features):
    test_others = pd.concat([test_others, read_parquet(f,index=False)], axis=1)

# ridge feature
ridge_features = glob.glob("../features/*ridge*train.parquet")
for f in sorted(ridge_features):
    if "stemmed_3_" in f or "tfidf_2" in f:
        continue
    train_others = pd.concat([train_others, read_parquet(f,index=False)], axis=1)
ridge_features = glob.glob("../features/*ridge*test.parquet")
for f in sorted(ridge_features):
    if "stemmed_3_" in f or "tfidf_2" in f:
        continue
    test_others = pd.concat([test_others, read_parquet(f,index=False)], axis=1)


# price features
# price_feat = glob.glob("../features/*price*train.parquet")
# for f in sorted(price_feat):
#     if "fe_item_price" in f or "fe_user_price" in f:
#         continue
#     train_others = pd.concat([train_others, read_parquet(f,index=False)], axis=1)
# price_feat = glob.glob("../features/*price*test.parquet")
# for f in sorted(price_feat):
#     if "fe_item_price" in f or "fe_user_price" in f:
#         continue
#     test_others = pd.concat([test_others, read_parquet(f,index=False)], axis=1)

train_features = pd.concat([read_parquet("../features/fe_price_isn_resampling_train.parquet"),
                            read_parquet("../features/fe_region_income_city_population_train.parquet")],axis=1)
test_features = pd.concat([read_parquet("../features/fe_price_isn_resampling_test.parquet"),
                                read_parquet("../features/fe_region_income_city_population_test.parquet")],axis=1)


X_train = pd.concat([X_train, train_others, train_features], axis=1)
X_test = pd.concat([X_test, test_others, test_features], axis=1)
X_train = X_train.drop(["user_id", "item_id"], axis=1)
X_test = X_test.drop(["user_id", "item_id"], axis=1)

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(X_train, categorica+["price+", "item_seq_number+"])
print(categorical_features_pos)

num_splits = 5
kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
val_predict = np.zeros(X_train.shape[0])
test_predict = np.zeros(X_test.shape[0])
X_train = X_train.values
X_test = X_test.values
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)

val_scores = []
for train_index, valid_index in kf.split(y):

    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]

    model = CatBoostRegressor(iterations=4000,
                             learning_rate=0.07,
                             depth=9,
                             #loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed=42,
                             od_type='Iter',
                             metric_period=100,
                             od_wait=200,
                             nan_mode="Min",
                             calc_feature_importance=False,
                             l2_leaf_reg=1)
    model.fit(X_train_fold, y_train_fold,
                 eval_set=(X_valid_fold, y_valid_fold),
                 cat_features=categorical_features_pos,
                 use_best_model=True,
                 verbose=True)

    val_predict[valid_index] = model.predict(X_valid_fold)
    test_predict += model.predict(X_test) / num_splits
    val_scores.append(rmse(y_valid_fold, model.predict(X_valid_fold)))

validation_score = np.mean(val_scores)
val_predicts = pd.DataFrame(data=val_predict, columns=["deal_probability"])
test_predict = pd.DataFrame(data=test_predict, columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/catboost_oof_yuki_val{}_train.csv'.format(validation_score), index=False)
test_predict.to_csv('../output/catboost_oof_yuki_val{}_test.csv'.format(validation_score), index=False)
print("ExtraTree Val Score: {}".format(validation_score))
notify_line("ExtraTree Val Score: {}".format(validation_score))
