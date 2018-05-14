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


from sklearn.utils import shuffle

debug = False

print("loading data ...")

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def feature_Eng_Datetime(df):
    print('feature engineering -> datetime ...')
    df['wday'] = df['activation_date'].dt.weekday
#    df['week'] = df['activation_date'].dt.week
#    df['dom'] = df['activation_date'].dt.day
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
    df['price'].fillna(-1, inplace=True)
    df.fillna('отсутствует описание', inplace=True) # google translation of 'missing discription' into Russian
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
    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv',  parse_dates = ["activation_date"])
#    
#    train_pr = pd.read_csv('../input/periods_train.csv',  parse_dates = ['activation_date', 'date_from', 'date_to'])
#    test_pr = pd.read_csv('../input/periods_test.csv',  parse_dates = ['activation_date', 'date_from', 'date_to'])
else: # debug
    train_df = pd.read_csv('../input/train.csv', parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234)
    train_df = train_df.iloc[:50000]
    y = train_df['deal_probability']
    del train_df['deal_probability']; gc.collect()
    test_df = pd.read_csv('../input/test.csv',  nrows=1000, parse_dates = ["activation_date"])
    
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



feature_Eng_Datetime(full_df)
feature_Eng_label_Enc(full_df)
feature_Eng_NA(full_df)
drop_image_data(full_df)

text_Hash(full_df)



#feature_Eng_time_pr(full_pr)
#print('merge train/test with interval info ...')
#full_df = full_df.join(full_pr, on="item_id", how='left', rsuffix='_')
#full_df.drop('item_id_', axis=1, inplace=True)
#full_df.fillna(-100000, inplace=True)

full_df.set_index('item_id', inplace=True)

# Meta Text Features
russian_stop = set(stopwords.words('russian'))
textfeats = ["description", "text_feat_p1_p2_p3", "text_feat_p2_p3", "title"]
for cols in textfeats:
    full_df[cols] = full_df[cols].astype(str).fillna('nicapotato').str.lower()
    full_df[cols + '_num_chars'] = full_df[cols].apply(len) # Count number of Characters
    full_df[cols + '_num_words'] = full_df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    full_df[cols + '_num_unique_words'] = full_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    full_df[cols + '_words_vs_unique'] = full_df[cols+'_num_unique_words'] / full_df[cols+'_num_words'] * 100 # Count Unique Words

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
            ngram_range=(1, 2),
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col('description'))),
                
        ('text_feat_p1_p2_p3',CountVectorizer(
            ngram_range=(1, 2),
            #max_features=7000,
            preprocessor=get_col('text_feat_p1_p2_p3'))),
#                
#        ('text_feat_p2_p3',CountVectorizer(
#            ngram_range=(1, 2),
#            #max_features=7000,
#            preprocessor=get_col('text_feat_p2_p3'))),
                
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            #max_features=7000,
            preprocessor=get_col('title')))
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

#print(full_df.info())
#del full_df
#gc.collect();


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
    'num_threads': 7,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 15,
    'num_leaves': 35,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.019,
    'verbose': 0,     
    # 'application': 'rmse',
    
#    'min_data':1,
#        'min_data_in_bin':100
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
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200,       
)
   
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
[16000]	train's rmse: 0.193696	valid's rmse: 0.219615
'''
