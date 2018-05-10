'''
训练集 3月15日到4月7日, 测试集4月12到4月20

'''

# TODO1: 使用period信息。
# TODO2: 计算各个类别物品的均值，最大值，平均值, 中值

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc

debug = True

print("loading data ...")

def feature_Eng_Datetime(df):
    print('feature engineering -> datetime ...')
    df['wday'] = df['activation_date'].dt.weekday
    df['week'] = df['activation_date'].dt.week
    df['dom'] = df['activation_date'].dt.day
    df.drop('activation_date', axis=1, inplace=True)
    return df

lbl = LabelEncoder()

def feature_Eng_label_Enc(df):
    print('feature engineering -> lable encoding ...')
    cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               # TODO: 这里还需要西考虑一下
               "param_1", "param_2", "param_3"]
    for col in cat_col:
        df[col] = lbl.fit_transform(df[col].astype(str))
    del cat_col;gc.collect()
    return df


def feature_Eng_NA(df):
    print('feature engineering -> handle NA ...')
    df['price'].fillna(-1, inplace=True)
    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian
    return df


#def feature_Eng_ON_price(df):
#    print('feature engineering -> statistics on price ...')
#    df['price'].fillna(-1, inplace=True)
#    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian
#    return df


def drop_image_data(df):
    print('feature engineering -> drop image data ...')
    df.drop('image', axis=1, inplace=True)
    return df
    
  
# load data
if debug == False: # Run
    train_df = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
    y = train_df['deal_probability']
    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
else: # debug
    train_df = pd.read_csv('../input/train.csv', index_col = "item_id", nrows=10000, parse_dates = ["activation_date"])
    y = train_df['deal_probability']
    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv', index_col = "item_id", nrows=10000, parse_dates = ["activation_date"])


train_index = len(train_df)
test_index = len(test_df)


# concat dataset
full_df = pd.concat([train_df, test_df], axis=0)
del train_df, test_df
gc.collect()



feature_Eng_Datetime(full_df)
feature_Eng_label_Enc(full_df)
feature_Eng_NA(full_df)
drop_image_data(full_df)


print(full_df.info())