# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(ML)s
"""

'''
训练集 3月15日到4月7日, 测试集4月12到4月20

'''

# TODO2: 计算各个类别物品的均值，最大值，平均值, 中值

from nltk.corpus import stopwords 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt


debug = False


print("loading data ...")

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def feature_Eng_Datetime(df):
    print('feature engineering -> datetime ...')
#    df['wday'] = df['activation_date'].dt.weekday
    df['week'] = df['activation_date'].dt.week
    df['dom'] = df['activation_date'].dt.day
    df['dow'] = df['activation_date'].dt.dayofweek
    df.drop('activation_date', axis=1, inplace=True)
    return df

lbl = LabelEncoder()
cat_col = ["user_id", "region", "city", "parent_category_name",
           "category_name", "user_type", "image_top_1",
           "param_1", "param_2", "param_3"]
def feature_Eng_label_Enc(df):
    print('feature engineering -> lable encoding ...')
    for col in cat_col:
        df[col] = lbl.fit_transform(df[col].astype(str))
    gc.collect()
    return df


def feature_Eng_NA(df):
    print('feature engineering -> handle NA ...')
    df['price'] = df['price'].fillna(12341)
    df['deal_probability'] = df['deal_probability'].fillna(12341)
#    df['price'] = np.log1p(df['price'])
#    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian
    return df


# =============================================================================
# 
# =============================================================================
# TODO: 仔细检查这个函数并且改进
def feature_Eng_On_Price_1(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['dow'], as_index=False)['price'].median().rename(columns={'price':'median_price_dow'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['dow'])
    df2['median_price_dow'] = df['median_price_dow']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_Price_2(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['param_1'], as_index=False)['price'].median().rename(columns={'price':'median_price_param_1'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['param_1'])
    df2['median_price_param_1'] = df['median_price_param_1']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_Price_3(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['city'], as_index=False)['price'].median().rename(columns={'price':'median_price_city'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['city'])
    df2['median_price_city'] = df['median_price_city']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_Price_4(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['category_name'], as_index=False)['price'].median().rename(columns={'price':'median_price_category_name'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['category_name'])
    df2['median_price_category_name'] = df['median_price_category_name']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_Price_5(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['image_top_1'], as_index=False)['price'].median().rename(columns={'price':'median_price_image_top_1'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['image_top_1'])
    df2['median_price_image_top_1'] = df['median_price_image_top_1']
    df2['price'] = np.log1p(df['price'])  
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_Price_A(df, df_train):
    print('feature engineering -> on price ...')
    tmp = df_train[df_train.price!=12341].groupby(['user_type'], as_index=False)['price'].median().rename(columns={'price':'median_price_user_type'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['user_type'])
    df2['median_price_user_type'] = df['median_price_user_type']
    df2['price'] = np.log1p(df['price'])  
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

#----------------------------------------------------------------------------------------------
    
def feature_Eng_On_deal_probability_1(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['dow'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_dow'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['dow'])
    df2['median_deal_probability_dow'] = df['median_deal_probability_dow']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2


def feature_Eng_On_deal_probability_2(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['param_1'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_param_1'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['param_1'])
    df2['median_deal_probability_param_1'] = df['median_deal_probability_param_1']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_deal_probability_3(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['city'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_city'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['city'])
    df2['median_deal_probability_city'] = df['median_deal_probability_city']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_deal_probability_4(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['category_name'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_category_name'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['category_name'])
    df2['median_deal_probability_category_name'] = df['median_deal_probability_category_name']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2


def feature_Eng_On_deal_probability_5(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['image_top_1'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_image_top_1'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['image_top_1'])
    df2['median_deal_probability_image_top_1'] = df['median_deal_probability_image_top_1']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

def feature_Eng_On_deal_probability_6(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['param_1', 'param_2', 'param_3'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_param_123'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['param_1', 'param_2', 'param_3'])
    df2['median_deal_probability_param_123'] = df['median_deal_probability_param_123']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2


def feature_Eng_On_deal_probability_A(df, df_train):
    print('feature engineering -> on deal_probability ...')
    tmp = df_train[df_train.deal_probability!=12341].groupby(['user_type'], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_user_type'}) 
    df2 = df
    df = pd.merge(df, tmp, how='left', on=['user_type'])
    df2['median_deal_probability_user_type'] = df['median_deal_probability_user_type']   
    df2.fillna(-1000, inplace=True)    
    del tmp; gc.collect()  
    return df2

# =============================================================================
# 
# =============================================================================

# TODO: 仔细检查这个函数并且改进
def feature_Eng_Hash(df):
    df['A'] = df['param_1'] * df['param_2']
    df['B'] = df['param_1'] * df['image_top_1']
    return df


def feature_Eng_time_pr(df):
    print('Feature engineering time data!')
    df['shelf_period'] = df['date_to'].dt.dayofyear - df['date_from'].dt.dayofyear
    df['waiting_period'] = df['date_from'].dt.dayofyear- df['activation_date'].dt.dayofyear
    df['total_period'] = df['date_to'].dt.dayofyear - df['activation_date'].dt.dayofyear
    df.drop(['activation_date', 'date_from', 'date_to'], axis=1, inplace=True)
    df.fillna(-10000, inplace=True)
    return df

def text_Hash(df):
    df['text_feat_p1_p2_p3'] = df.apply(lambda row: ' '.join([
            str(row['param_1']), str(row['param_2']), str(row['param_3'])
            ]),axis=1)
    df['text_feat_p2_p3'] = df.apply(lambda row: ' '.join([
            str(row['param_2']), str(row['param_3'])
        ]),axis=1)
    return df
    

def drop_image_data(df):
    print('feature engineering -> drop image data ...')
    df.drop('image', axis=1, inplace=True)
    return df
    
  
# load data
if debug == False: # Run
    train_df = pd.read_csv('../input/train.csv',  parse_dates = ["activation_date"])
    y = train_df['deal_probability']
#    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv',  parse_dates = ["activation_date"])
    
#    train_pr = pd.read_csv('../input/periods_train.csv',  parse_dates = ['activation_date', 'date_from', 'date_to'])
#    test_pr = pd.read_csv('../input/periods_test.csv',  parse_dates = ['activation_date', 'date_from', 'date_to'])
else: # debug
    train_df = pd.read_csv('../input/train.csv', nrows=50000, parse_dates = ["activation_date"])
    y = train_df['deal_probability']
#    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv',  nrows=50000, parse_dates = ["activation_date"])
    
#    train_pr = pd.read_csv('../input/periods_train.csv',  nrows=10000, parse_dates = ['activation_date', 'date_from', 'date_to'])
#    test_pr = pd.read_csv('../input/periods_test.csv',  nrows=10000, parse_dates = ['activation_date', 'date_from', 'date_to'])


train_index = len(train_df)
test_index = test_df['item_id']


# concat dataset
full_df = pd.concat([train_df, test_df], axis=0)
del train_df, test_df
gc.collect()

#full_pr = pd.concat([train_pr, test_pr], axis=0)
#del train_pr, test_pr
#gc.collect()

print(len(list(full_df)), list(full_df))

feature_Eng_Datetime(full_df)
# =============================================================================
# 
# =============================================================================
feature_Eng_NA(full_df)

X_train, X_valid= train_test_split(full_df.iloc[:train_index], test_size=0.10, random_state=23)

feature_Eng_On_Price_1(full_df, X_train)
feature_Eng_On_Price_2(full_df, X_train)
feature_Eng_On_Price_3(full_df, X_train)
feature_Eng_On_Price_4(full_df, X_train)
feature_Eng_On_Price_5(full_df, X_train)
#feature_Eng_On_Price_A(full_df, X_train)

# mean probability
feature_Eng_On_deal_probability_1(full_df, X_train)
feature_Eng_On_deal_probability_2(full_df, X_train)
feature_Eng_On_deal_probability_3(full_df, X_train)
feature_Eng_On_deal_probability_4(full_df, X_train)
feature_Eng_On_deal_probability_5(full_df, X_train)
feature_Eng_On_deal_probability_6(full_df, X_train)
#feature_Eng_On_deal_probability_A(full_df, X_train)
del X_train, X_valid

gc.collect()


feature_Eng_label_Enc(full_df)

#feature_Eng_Hash(full_df)
drop_image_data(full_df)

del full_df['deal_probability']; gc.collect()

#text_Hash(full_df)
# =============================================================================
# 
# =============================================================================



full_df.set_index('item_id', inplace=True)




# TODO: 考虑要不要使用这个
#feature_Eng_time_pr(full_pr)
#print('merge train/test with interval info ...')
#full_df = full_df.join(full_pr, on="item_id", how='left', rsuffix='_')
#full_df.drop('item_id_', axis=1, inplace=True)
#full_df.fillna(-100000, inplace=True)




# Meta Text Features
russian_stop = set(stopwords.words('russian'))
#textfeats = ["description", "text_feat_p1_p2_p3", "text_feat_p2_p3", "title"]
textfeats = ["description",  "title"]
for cols in textfeats:
    full_df[cols] = full_df[cols].astype(str).fillna('nicapotato').str.lower()
    full_df[cols + '_num_chars'] = full_df[cols].apply(len) # Count number of Characters
    full_df[cols + '_num_words'] = full_df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    full_df[cols + '_num_unique_words'] = full_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    full_df[cols + '_words_vs_unique'] = full_df[cols+'_num_unique_words'] / full_df[cols+'_num_words'] * 100 # Count Unique Words

# word
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2), #(1,2)
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col('description'))),
                
#        ('text_feat_p1_p2_p3',CountVectorizer(
#            ngram_range=(1, 2), #(1,2)
#            #max_features=7000,
##            **tfidf_para,
#            preprocessor=get_col('text_feat_p1_p2_p3'))),
                                
        ('title',TfidfVectorizer(
            ngram_range=(1, 2), #(1,2)
            **tfidf_para,
            #max_features=7000,
            preprocessor=get_col('title'))),
        
    ])
vectorizer.fit(full_df.to_dict('records'))
ready_full_df = vectorizer.transform(full_df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()

full_df.drop(textfeats, axis=1,inplace=True)
#full_df.drop(['param_1','param_2','param_3'], axis=1,inplace=True)
full_df.fillna(-10000, inplace=True)



print("Modeling Stage ...")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(full_df.iloc[:train_index]), ready_full_df[:train_index]]) # Sparse Matrix
test = hstack([csr_matrix(full_df.iloc[train_index:]), ready_full_df[train_index:]]) # Sparse Matrix

tfvocab = full_df.columns.tolist() + tfvocab
for shape in [X,test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))

print(full_df.info())
#del full_df
gc.collect();


cat_col = [
           "user_id",
           "region", 
           "city", 
           "parent_category_name",
           "category_name", 
           "user_type", 
           "image_top_1",
           "param_1", 
           "param_2", 
           "param_3",
           ]

# Begin trainning


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)
    

lgbm_params =  {
    
#    'device' : 'gpu',
#    'gpu_platform_id' : -1,
#    'gpu_device_id' : -1,
    'tree_method': 'feature',
    
    'num_threads': 3,

    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'rmse',
    'metric': 'rmse',
    'max_depth': 15,
#    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.99,
    'bagging_freq': 10,
    'learning_rate': 0.019,
    'verbose': 0, 

} 

lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = cat_col)

lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = cat_col)

lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=32000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=500,       
)

# Save model
from sklearn.externals import joblib
print('save model ...')
joblib.dump(lgb_clf, 'lgb.pkl')
# load model
#lgb_clf = joblib.load('lgb.pkl')

gc.collect()

   
print("Model Evaluation Stage")
# TODO: 这一行为什么算出来的不一样?
print('RMSE:', np.sqrt(mean_squared_error(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)) ))
lgpred = lgb_clf.predict(test, num_iteration=lgb_clf.best_iteration)


lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=test_index)
lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
lgsub.to_csv("lgsub.csv",index=True,header=True)


print("Features importance...")
bst = lgb_clf
gain = bst.feature_importance('gain')
ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(50))

plt.figure()
ft[['feature','gain']].head(50).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 20))
plt.gcf().savefig('features_importance.png')

print("Done.")


'''
n_gram (1, 5)
[15589] train's rmse: 0.194768  valid's rmse: 0.219945


[15343] train's rmse: 0.194989  valid's rmse: 0.219975
'''
