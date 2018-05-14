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
import gc, re
import datetime as dt
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from math import sqrt
from contextlib import contextmanager

debug = True
print("loading data ...")
if debug == False:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    del train_df["deal_probability"]; gc.collect()
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])
else:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:50000]
    y = train_df["deal_probability"]
    del train_df["deal_probability"]; gc.collect()
    test_df = pd.read_csv("../input/test.csv",  nrows=1000, parse_dates = ["activation_date"])

def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def text_preprocessing(text):        
    text = str(text)
    text = text.lower() 
    # hash words
    text = re.sub(r"(\\u[0-9A-Fa-f]+)",r"", text)
    return text

@contextmanager
def feature_engineering(df):
    # All the feature engineering here
    
    def Do_Text_Hash(df):
        print("feature engineering -> hash text ...")
        df["text_feature"] = df.apply(lambda row: " ".join([str(row["param_1"]),
          str(row["param_2"]), str(row["param_3"])]),axis=1)
    
        df["text_feature_2"] = df.apply(lambda row: " ".join([str(row["param_2"]), str(row["param_3"])]),axis=1)
        
        df["title_description"] = df.apply(lambda row: " ".join([str(row["title"]), str(row["description"])]),axis=1)
       
        print("feature engineering -> preprocess text ...")       
        df["text_feature"] = df["text_feature"].apply(lambda x: text_preprocessing(x))
        df["text_feature_2"] = df["text_feature_2"].apply(lambda x: text_preprocessing(x))
        df["description"] = df["description"].apply(lambda x: text_preprocessing(x))
        df["title"] = df["title"].apply(lambda x: text_preprocessing(x))
        df["title_description"] = df["title_description"].apply(lambda x: text_preprocessing(x))
        
                 
    def Do_Datetime(df):
        print("feature engineering -> date time ...")
        df["wday"] = df["activation_date"].dt.weekday
#        df["week"] = df["activation_date"].dt.week
#        df["dom"] = df["activation_date"].dt.day
        
    def Do_Label_Enc(df):
        print("feature engineering -> lable encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               "param_1", "param_2", "param_3", "image"]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            
    def Do_NA(df):
        print("feature engineering -> fill na ...")
#        df["price"] = np.log(df["price"]+0.001).astype("float32")
        df["price"].fillna(-1,inplace=True)
        df["image_top_1"].fillna("nicapotato",inplace=True)
        df["image"].fillna("noinformation",inplace=True)
        df["param_1"].fillna("nicapotato",inplace=True)
        df["param_2"].fillna("nicapotato",inplace=True)
        df["param_3"].fillna("nicapotato",inplace=True)
        df["title"].fillna("nicapotato",inplace=True)
        df["description"].fillna("nicapotato",inplace=True)
             
    def Do_Drop(df):
        df.drop(["activation_date", "item_id"], axis=1, inplace=True)
        
    def Do_Stat_Text(df):
        print("feature engineering -> statistics in text ...")
        textfeats = ["text_feature","text_feature_2","description", "title"]
        for col in textfeats:
            df[col + "_num_chars"] = df[col].apply(len) 
            df[col + "_num_words"] = df[col].apply(lambda comment: len(comment.split()))
            df[col + "_num_unique_words"] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
            df[col + "_words_vs_unique"] = df[col+"_num_unique_words"] / df[col+"_num_words"] * 100
                      
    # choose which functions to run
    Do_NA(df)
    Do_Text_Hash(df)    
    Do_Datetime(df)
    Do_Label_Enc(df)
    Do_Stat_Text(df)
       
    Do_Drop(df)    
    gc.collect()
    return df

def data_vectorize(df):
    russian_stop = set(stopwords.words("russian"))
    tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": "word",
    "token_pattern": r"\w{1,}",
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": "l2",
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
    }
    def get_col(col_name): return lambda x: x[col_name]
    vectorizer = FeatureUnion([
            
            ("description",TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=18000,
                    **tfidf_para,
                    preprocessor=get_col("description"))
                ),
    
            ("title_description",TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=18000,
                    **tfidf_para,
                    preprocessor=get_col("title_description"))
                ),
    
            ("text_feature",CountVectorizer(
                    ngram_range=(1, 2),
                    preprocessor=get_col("text_feature"))
                ),
        
            ("title",TfidfVectorizer(
                    ngram_range=(1, 2),
                    **tfidf_para,
                    preprocessor=get_col("title"))
                ),
    ])
    
    vectorizer.fit(df.to_dict("records"))
    ready_full_df = vectorizer.transform(df.to_dict("records"))
    tfvocab = vectorizer.get_feature_names()
    
    df.drop(["text_feature", "text_feature_2", "description","title", "title_description"], axis=1, inplace=True)
    df.fillna(-1, inplace=True)
     
    return df, ready_full_df, tfvocab

full_df = pd.concat([train_df, test_df])
sub_item_id = test_df["item_id"]
len_train = len(train_df)
del train_df, test_df; gc.collect()

feature_engineering(full_df)
full_df, ready_full_df, tfvocab = data_vectorize(full_df)

# --------------------------------------------------------------------------------------
print("Modeling Stage ...")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
test = hstack([csr_matrix(full_df.iloc[len_train:]), ready_full_df[len_train:]]) # Sparse Matrix
tfvocab = full_df.columns.tolist() + tfvocab

for shape in [X,test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))

cat_col = [
#           "user_id",
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
        'tree_method': 'feature',    
        'num_threads': 3,
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
#        'application': 'rmse',
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
print( 'RMSE:', rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)) )
lgpred = lgb_clf.predict(test, num_iteration=lgb_clf.best_iteration)

lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=sub_item_id)
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