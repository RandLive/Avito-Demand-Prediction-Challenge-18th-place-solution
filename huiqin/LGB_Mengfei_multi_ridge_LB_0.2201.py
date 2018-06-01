#encoding=utf-8
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc, re
from sklearn.utils import shuffle
from contextlib import contextmanager
from sklearn.externals import joblib
import nltk
# nltk.download('stopwords')
import time
print("Starting job at time:",time.time())
debug =False#False,True
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])
    # suppl
    train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=["date_from", "date_to"])
else:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    train_index=5000*2
    row_count=1000*2
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:train_index]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  nrows=row_count, parse_dates = ["activation_date"])
    # suppl 
    train_active = pd.read_csv("../input/train_active.csv",  nrows=row_count, usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv",  nrows=row_count, usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv",  nrows=row_count, parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv",  nrows=row_count, parse_dates=["date_from", "date_to"])
print("loading data done!")

# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/region_income.csv", sep=";", names=["region", "income"])
#tmp["income"] = np.log1p(tmp["income"])
train_df = train_df.merge(tmp, on="region", how="left")
test_df = test_df.merge(tmp, on="region", how="left")
del tmp; gc.collect()
# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/city_population_wiki_v3.csv")
#tmp["population"] = np.log1p(tmp["population"])
train_df = train_df.merge(tmp, on="city", how="left")
test_df = test_df.merge(tmp, on="city", how="left")
del tmp; gc.collect()

# =============================================================================
# Here Based on https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/code
# =============================================================================
all_samples = pd.concat([train_df,train_active,test_df,test_active]).reset_index(drop=True)
all_samples.drop_duplicates(["item_id"], inplace=True)
del train_active, test_active; gc.collect()

all_periods = pd.concat([train_periods,test_periods])
del train_periods, test_periods; gc.collect()

all_periods["days_up"] = (all_periods["date_to"] - all_periods["date_from"]).dt.days
gp = all_periods.groupby(["item_id"])[["days_up"]]

gp_df = pd.DataFrame()
gp_df["days_up_sum"] = gp.sum()["days_up"]
gp_df["times_put_up"] = gp.count()["days_up"]
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={"index": "item_id"})

all_periods.drop_duplicates(["item_id"], inplace=True)
all_periods = all_periods.merge(gp_df, on="item_id", how="left")
all_periods = all_periods.merge(all_samples, on="item_id", how="left")

gp = all_periods.groupby(["user_id"])[["days_up_sum", "times_put_up"]].mean().reset_index()\
.rename(index=str, columns={"days_up_sum": "avg_days_up_user",
                            "times_put_up": "avg_times_up_user"})

n_user_items = all_samples.groupby(["user_id"])[["item_id"]].count().reset_index() \
.rename(index=str, columns={"item_id": "n_user_items"})
gp = gp.merge(n_user_items, on="user_id", how="outer") #left

del all_samples, all_periods, n_user_items
gc.collect()

train_df = train_df.merge(gp, on="user_id", how="left")
test_df = test_df.merge(gp, on="user_id", how="left")

agg_cols = list(gp.columns)[1:]

del gp; gc.collect()

for col in agg_cols:
    train_df[col].fillna(-1, inplace=True)
    test_df[col].fillna(-1, inplace=True)

print("merging supplimentary data done!")


# =============================================================================
# done! go to the normal steps
# =============================================================================
def rmse(predictions, targets):
    print("calculating RMSE ...")
    return np.sqrt(((predictions - targets) ** 2).mean())

def text_preprocessing(text):        
    text = str(text)
    text = text.lower()
    # 去掉一些特殊符号

    # text = re.sub(r'\"', ' ', text)
    # text = re.sub(r'\n', ' ', text)
    # text = re.sub(r'\t', ' ', text)
    # text = re.sub(r'\:', ' ', text)
    # text = re.sub(r'\•', ' ', text)
    # # string = re.sub(r'_', ' ', string)
    # text = re.sub(r'\+', ' ', text)
    # text = re.sub(r'\=', ' ', text)
    # text = re.sub(r'\,', ' ', text)
    # text = re.sub(r'\.', ' ', text)
    #
    # text = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', text)
    # hash words
    text = re.sub(r"(\\u[0-9A-Fa-f]+)",r"", text)
    
    text = re.sub(r"===",r" ", text)
    
    # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = " ".join(text.split())
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
        df["wday"] =df["wday"].astype(np.uint8)
#        df["week"] = df["activation_date"].dt.week
#        df["dom"] = df["activation_date"].dt.day
        
    def Do_Label_Enc(df):
        print("feature engineering -> lable encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               "param_1", "param_2", "param_3","image"]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()
    
    import string
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])         
    def Do_NA(df):
        print("feature engineering -> fill na ...")
#        df["price"] = np.log(df["price"]+0.001).astype("float32")
#        df["price"].fillna(-1,inplace=True)
        df["image_top_1"].fillna(-1,inplace=True)
#        df["image_top_4"].fillna(-1,inplace=True)
        df["image"].fillna("noinformation",inplace=True)
        df["param_1"].fillna("nicapotato",inplace=True)
        df["param_2"].fillna("nicapotato",inplace=True)
        df["param_3"].fillna("nicapotato",inplace=True)
        df["title"].fillna("nicapotato",inplace=True)
        df["description"].fillna("nicapotato",inplace=True)
        
    def Do_Count(df):  
        print("feature engineering -> do count ...")
        # some count       
        df["num_desc_punct"] = df["description"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
        df["num_desc_capE"] = df["description"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
        df["num_desc_capP"] = df["description"].apply(lambda x: count(x, "[А-Я]")).astype(np.int16)
        
        df["num_title_punct"] = df["title"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
        df["num_title_capE"] = df["title"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
        df["num_title_capP"] = df["title"].apply(lambda x: count(x, "[А-Я]"))  .astype(np.int16)
        # good, used, bad ... count
        df["is_in_desc_хорошо"] = df["description"].str.contains("хорошо").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_Плохо"] = df["description"].str.contains("Плохо").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_новый"] = df["description"].str.contains("новый").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_старый"] = df["description"].str.contains("старый").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_используемый"] = df["description"].str.contains("используемый").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_есплатная_доставка"] = df["description"].str.contains("есплатная доставка").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_есплатный_возврат"] = df["description"].str.contains("есплатный возврат").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_идеально"] = df["description"].str.contains("идеально").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_подержанный"] = df["description"].str.contains("подержанный").map({True:1, False:0}).astype(np.uint8)
        df["is_in_desc_пСниженные_цены"] = df["description"].str.contains("Сниженные цены").map({True:1, False:0}).astype(np.uint8)
        
                              
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
            gc.collect()
                      
    # choose which functions to run
    Do_NA(df)
    Do_Text_Hash(df)
    Do_Label_Enc(df)
    Do_Count(df)
    Do_Datetime(df)   
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

    tfidf_para2 = {
        "stop_words": russian_stop,
        "analyzer": "char",
        "token_pattern": r"\w{1,}",
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": "l2",
        # "min_df":5,
        # "max_df":.9,
        "smooth_idf": False
    }

    def get_col(col_name): return lambda x: x[col_name]
    vectorizer = FeatureUnion([
        ("description", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,#40000,18000
            **tfidf_para,
            preprocessor=get_col("description"))
         ),
        # ("title_description", TfidfVectorizer(
        #      ngram_range=(1, 2),#(1,2)
        #      max_features=8000,#40000,18000
        #      **tfidf_para,
        #      preprocessor=get_col("title_description"))
        #   ),
        ("text_feature", CountVectorizer(
            ngram_range=(1, 2),
            preprocessor=get_col("text_feature"))
         ),
        ("title", TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            preprocessor=get_col("title"))
         ),
        #新加入两个文本处理title2，title_char
        ("title2", TfidfVectorizer(
            ngram_range=(1, 1),
            **tfidf_para,
            preprocessor=get_col("title"))
         ),

        ("title_char", TfidfVectorizer(

            ngram_range=(1, 4),#(1, 4),(1,6)
            max_features=16000,#16000
            **tfidf_para2,
            preprocessor=get_col("title"))
         ),
    ])
    vectorizer.fit(df.to_dict("records"))
    ready_full_df = vectorizer.transform(df.to_dict("records"))    
    tfvocab = vectorizer.get_feature_names()    
    df.drop(["text_feature", "text_feature_2", "description","title", "title_description"], axis=1, inplace=True)
    df.fillna(-1, inplace=True)     
    return df, ready_full_df, tfvocab
# =============================================================================
# Ridge feature https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
# =============================================================================
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

NFOLDS = 10#5
SEED = 42
def get_oof(clf, x_train, y, x_test):
            
    oof_train = np.zeros((len_train,))
    oof_test = np.zeros((len_test,))
    oof_test_skf = np.empty((NFOLDS, len_test))

    for i, (train_index, test_index) in enumerate(kf):
        print('Ridege oof Fold {}'.format(i))
        x_tr = x_train[train_index]       
        y = np.array(y)
        y_tr = y[train_index]
        x_te = x_train[test_index]      
        clf.train(x_tr, y_tr)       
        oof_train[test_index] = clf.predict(x_te)        
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


full_df = pd.concat([train_df, test_df])
sub_item_id = test_df["item_id"]
len_train = len(train_df)
len_test = len(test_df)

kf = KFold(len_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)
# =============================================================================
# handle price
# =============================================================================
def feature_Eng_On_Price_SEQ(df):
    print('feature engineering -> on price and SEQ ...')        
    df["price"] = np.log(df["price"]+0.001).astype("float32")
    df["price"].fillna(-1,inplace=True)     
    df["price+"] = np.round(df["price"]*4.8).astype(np.int16)
    df["item_seq_number+"] = np.round(df["item_seq_number"]/100).astype(np.int16)
    return df

train_df, val_df = train_test_split(
    full_df.iloc[:len_train], test_size=0.1, random_state=42) #23    
def feature_Eng_On_Deal_Prob(df, df_train):
    print('feature engineering -> on price deal prob +...')
    df2 = df    
    tmp = df_train.groupby(["price+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_price+'})     
    df = pd.merge(df, tmp, how='left', on=["price+"])
    df2['median_deal_probability_price+'] = df['median_deal_probability_price+']
    df2['median_deal_probability_price+'] =df2['median_deal_probability_price+'].astype(np.float32)
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["item_seq_number+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_item_seq_number+'})     
    df = pd.merge(df, tmp, how='left', on=["item_seq_number+"])
    df2['median_deal_probability_item_seq_number+'] = df['median_deal_probability_item_seq_number+']
    df2['median_deal_probability_item_seq_number+'] =df2['median_deal_probability_item_seq_number+'].astype(np.float32)
       
    df2.fillna(-1, inplace=True)    
    del tmp; gc.collect()
    return df2

del full_df['deal_probability']; gc.collect()

# =============================================================================
# use additianl image data
# =============================================================================
feature_engineering(full_df)

feature_Eng_On_Price_SEQ(full_df)
feature_Eng_On_Price_SEQ(train_df)
feature_Eng_On_Deal_Prob(full_df, train_df)

del train_df, test_df; gc.collect()
full_df, ready_full_df, tfvocab = data_vectorize(full_df)

#'alpha':20.0
ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ready_df = ready_full_df

print('ridge 1 oof ...')
ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(full_df)[:len_train], y, np.array(full_df)[len_train:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
full_df['ridge_preds_1'] = ridge_preds
full_df['ridge_preds_1'].clip(0.0, 1.0, inplace=True)

print('ridge 2 oof ...')
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
full_df['ridge_preds_2'] = ridge_preds
full_df['ridge_preds_2'].clip(0.0, 1.0, inplace=True)
del ridge_oof_train, ridge_oof_test,ridge_preds,ridge,ready_df
gc.collect()

print("Modeling Stage ...")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
# test = hstack([csr_matrix(full_df.iloc[len_train:]), ready_full_df[len_train:]]) # Sparse Matrix
tfvocab = full_df.columns.tolist() + tfvocab
X_test_full=full_df.iloc[len_train:]
X_test_ready=ready_full_df[len_train:]
del ready_full_df,full_df
gc.collect()
#
# for shape in [X,test]:
#     print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))

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
           "price+",
           "item_seq_number+",          
           ]

rmse_sume = 0.

for numIter in range(0, 1):
      
      X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42) #23
      
#      X_train, X_valid = X.tocsr()[train_index], X.tocsr()[test_index]
#      y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
      del X,y
      gc.collect()
      lgbm_params =  {
              "tree_method": "feature",    
              "num_threads": 7,
              "task": "train",
              "boosting_type": "gbdt",
              "objective": "regression",
              "metric": "rmse",
             # "max_depth": 15,
              "num_leaves": 500, # 280,360,500,32
              "feature_fraction": 0.2, #0.4
              "bagging_fraction": 0.2, #0.4
              "learning_rate": 0.015,#0.015
              "verbose": -1,
              'lambda_l1':1,
              'lambda_l2':1,
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
              valid_names=["train","valid"],
              early_stopping_rounds=200,
              verbose_eval=100, #200
              )
      
      print("save model ...")
      joblib.dump(lgb_clf, "lgb_{}.pkl".format(numIter))
      ## load model
      #lgb_clf = joblib.load("lgb.pkl")
      
      print("Model Evaluation Stage")
      print( "RMSE:", rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)) )

      test = hstack([csr_matrix(X_test_full), X_test_ready])  # Sparse Matrix
      lgpred = lgb_clf.predict(test, num_iteration=lgb_clf.best_iteration)
      
      lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=sub_item_id)
      lgsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
      lgsub.to_csv("ml_lgb_sub_{}.csv".format(numIter),index=True,header=True)

      rmse_sume += rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration))
      
      numIter += 1
      
      del X_train, X_valid, y_train, y_valid, lgtrain, lgvalid
      gc.collect()

print("mean rmse is:", rmse_sume/5)
      
print("Features importance...")
bst = lgb_clf
gain = bst.feature_importance("gain")
ft = pd.DataFrame({"feature":bst.feature_name(), "split":bst.feature_importance("split"), "gain":100 * gain / gain.sum()}).sort_values("gain", ascending=False)
print(ft.head(50))

plt.figure()
ft[["feature","gain"]].head(50).plot(kind="barh", x="feature", y="gain", legend=False, figsize=(10, 20))
plt.gcf().savefig("features_importance.png")

print("Done.")


"""

Training until validation scores don't improve for 200 rounds.
[100]   train's rmse: 0.222137  valid's rmse: 0.223204
[200]   train's rmse: 0.218536  valid's rmse: 0.22062
[300]   train's rmse: 0.216667  valid's rmse: 0.219558
[400]   train's rmse: 0.215363  valid's rmse: 0.218984
[500]   train's rmse: 0.214371  valid's rmse: 0.218628
[600]   train's rmse: 0.213575  valid's rmse: 0.218439
[700]   train's rmse: 0.212943  valid's rmse: 0.218287
[800]   train's rmse: 0.212364  valid's rmse: 0.218184
[900]   train's rmse: 0.211831  valid's rmse: 0.21809
[1000]  train's rmse: 0.211392  valid's rmse: 0.218054
[1100]  train's rmse: 0.210915  valid's rmse: 0.217974
[1200]  train's rmse: 0.210504  valid's rmse: 0.217943
[1300]  train's rmse: 0.210084  valid's rmse: 0.217897
[1400]  train's rmse: 0.209658  valid's rmse: 0.217857
[1500]  train's rmse: 0.209265  valid's rmse: 0.217817
[1600]  train's rmse: 0.20888   valid's rmse: 0.217795
[1700]  train's rmse: 0.20849   valid's rmse: 0.217757
[1800]  train's rmse: 0.208115  valid's rmse: 0.217727
[1900]  train's rmse: 0.207731  valid's rmse: 0.217703
[2000]  train's rmse: 0.207364  valid's rmse: 0.217665
[2100]  train's rmse: 0.206997  valid's rmse: 0.217619
[2200]  train's rmse: 0.206597  valid's rmse: 0.217602


[6200]	train's rmse: 0.195332	valid's rmse: 0.216305
[6300]	train's rmse: 0.195234	valid's rmse: 0.216305
[6400]	train's rmse: 0.195118	valid's rmse: 0.216301
[6500]	train's rmse: 0.195007	valid's rmse: 0.216297

Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 0.224907	valid's rmse: 0.227163
[200]	train's rmse: 0.217509	valid's rmse: 0.221793
[300]	train's rmse: 0.213607	valid's rmse: 0.219756
[400]	train's rmse: 0.210737	valid's rmse: 0.218695
[500]	train's rmse: 0.208375	valid's rmse: 0.217965

mean rmse is: 0.21697176669518864

"num_leaves": 32,
"learning_rate": 0.05
Training until validation scores don't improve for 200 rounds.
[100]   train's rmse: 0.181808  valid's rmse: 0.241621
[200]   train's rmse: 0.148168  valid's rmse: 0.243786
Early stopping, best iteration is:
[67]    train's rmse: 0.197385  valid's rmse: 0.24111

Early stopping, best iteration is:
[199]	train's rmse: 0.179879	valid's rmse: 0.232363

Early stopping, best iteration is:
[207]	train's rmse: 0.176758	valid's rmse: 0.232526

ridge alpha=4
Early stopping, best iteration is:
[192]   train's rmse: 0.180265  valid's rmse: 0.232168



Early stopping, best iteration is:
[200]	train's rmse: 0.178794	valid's rmse: 0.232766

Early stopping, best iteration is:
[190]	train's rmse: 0.181175	valid's rmse: 0.232942

Early stopping, best iteration is:
[194]	train's rmse: 0.180253	valid's rmse: 0.233221

Early stopping, best iteration is:学习率0.1
[28]    train's rmse: 0.20511   valid's rmse: 0.242049

Early stopping, best iteration is:
[243]   train's rmse: 0.194437  valid's rmse: 0.241955

Early stopping, best iteration is:(1,8)
[150]   train's rmse: 0.193472  valid's rmse: 0.242612

Early stopping, best iteration is:
[379]	train's rmse: 0.145653	valid's rmse: 0.241053


Early stopping, best iteration is:
[333]   train's rmse: 0.152137  valid's rmse: 0.241698

Early stopping, best iteration is:(1,6)
[333]   train's rmse: 0.152137  valid's rmse: 0.241698

[178]   train's rmse: 0.190238  valid's rmse: 0.244317(1,1)

Early stopping, best iteration is: (1,2)
[320]   train's rmse: 0.155529  valid's rmse: 0.242601

Early stopping, best iteration is:(1,4)
[215]	train's rmse: 0.17696	valid's rmse: 0.242142

[267]	train's rmse: 0.183306	valid's rmse: 0.243442

Early stopping, best iteration is:
[193]	train's rmse: 0.199016	valid's rmse: 0.244293

Early stopping, best iteration is:
[271]	train's rmse: 0.183924	valid's rmse: 0.243879

Early stopping, best iteration is:
[196]	train's rmse: 0.197641	valid's rmse: 0.244071

Early stopping, best iteration is:
[204]	train's rmse: 0.195703	valid's rmse: 0.243714
"""
