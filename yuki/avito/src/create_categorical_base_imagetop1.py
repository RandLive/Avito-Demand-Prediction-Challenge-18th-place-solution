# category: ["region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3"]("user_id"?)
# 1. category base count features
# 2. category embedding.
from utils import *
import pandas as pd
import gc
train = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"]).drop(["deal_probability", "image"],axis=1)
test = pd.read_csv("../input/test.csv", parse_dates = ["activation_date"]).drop(["image"],axis=1)
all_df = pd.concat([train, test])
all_df["dayofweek"] = all_df.activation_date.dt.weekday
train["dayofweek"] = train.activation_date.dt.weekday
test["dayofweek"] = test.activation_date.dt.weekday
all_df["one"] = 1
print("Done Reading All Data")

def get_category_df(col):
    tmp = pd.DataFrame()
    all_df[col] = all_df[col].fillna("NAN")
    tmp["{}_count".format(col)] = all_df.groupby(col)["one"].sum()
    tmp["{}_unique_user_count".format(col)] = all_df[["user_id", col]].groupby(col).agg(pd.Series.nunique)
    tmp["{}_price_median".format(col)] = all_df[[col, "price"]].groupby(col).agg(np.median)
    tmp["{}_price_std".format(col)] = all_df[[col, "price"]].groupby(col).agg(np.std)
    tmp["{}_price_max".format(col)] = all_df[[col, "price"]].groupby(col).agg(np.max)
    tmp["{}_price_min".format(col)] = all_df[[col, "price"]].groupby(col).agg(np.min)
    tmp["latest_date"] = all_df[[col, "activation_date"]].groupby(col).max()
    tmp["first_date"] = all_df[[col, "activation_date"]].groupby(col).min()
    tmp["{}_diff".format(col)] = (tmp["latest_date"] - tmp["first_date"]).dt.days
    tmp["{}_average_period".format(col)] = tmp["{}_diff".format(col)] / tmp["{}_count".format(col)]
    tmp.drop(["latest_date", "first_date"], axis=1, inplace=True)

    return tmp.reset_index()

print("Categorical Features...")
image_top_1 = get_category_df("image_top_1")


category = {"image_top_1":image_top_1}
cate_col = list(category.keys())
train = train[cate_col]
test = test[cate_col]
for col, d in category.items():
    train = pd.merge(train, d, on=col, how="left")
    test = pd.merge(test, d, on=col, how="left")

train.drop(cate_col, axis=1, inplace=True)
test.drop(cate_col, axis=1, inplace=True)
to_parquet(train, "../features/fe_categorical_base_2_features_train.parquet")
to_parquet(test, "../features/fe_categorical_base_2_features_test.parquet")

# weekday
def get_category_weekday_df(col):
    all_df[col] = all_df[col].fillna("NAN")
    tmp = pd.DataFrame()
    tmp["{}_{}_count".format(*col)] = all_df.groupby(col)["one"].sum()
    tmp["{}_{}_unique_user_count".format(*col)] = all_df[["user_id"] + col].groupby(col).agg(pd.Series.nunique)
    tmp["{}_{}_price_median".format(*col)] = all_df[["price"] + col].groupby(col).agg(np.median)
    tmp["{}_{}_price_std".format(*col)] = all_df[["price"] + col].groupby(col).agg(np.std)
    tmp["{}_{}_price_max".format(*col)] = all_df[["price"] + col].groupby(col).agg(np.max)
    tmp["{}_{}_price_min".format(*col)] = all_df[["price"] + col].groupby(col).agg(np.min)
    tmp = tmp.reset_index()
    return tmp

print("Categorical Weekday Features...")
image_top_1 = get_category_weekday_df(["image_top_1", "dayofweek"])

category = {"image_top_1":image_top_1}
cate_col = list(category.keys())+["dayofweek"]
train = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"]).drop(["deal_probability", "image"],axis=1)
test = pd.read_csv("../input/test.csv", parse_dates = ["activation_date"]).drop(["image"],axis=1)
train["dayofweek"] = train.activation_date.dt.weekday
test["dayofweek"] = test.activation_date.dt.weekday
train = train[cate_col]
test = test[cate_col]
for col, d in category.items():
    train = pd.merge(train, d, on=[col, "dayofweek"], how="left")
    test = pd.merge(test, d, on=[col, "dayofweek"], how="left")

train.drop(cate_col, axis=1, inplace=True)
test.drop(cate_col, axis=1, inplace=True)

to_parquet(train, "../features/fe_categorical_weekday_2_features_train.parquet")
to_parquet(test, "../features/fe_categorical_weekday_2_features_test.parquet")
