#encoding=utf-8
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
import lightgbm as lgb
#import matplotlib.pyplot as plt
import gc, re
from sklearn.utils import shuffle
from contextlib import contextmanager
from sklearn.externals import joblib
import time

print("Starting job at time:",time.time())
debug = True
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("../input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  parse_dates = ["activation_date"])

    train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=["date_from", "date_to"])
else:
    train_df = pd.read_csv("../input/train.csv", parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:10000]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  nrows=1000, parse_dates = ["activation_date"])
    
    train_active = pd.read_csv("../input/train_active.csv",  nrows=1000, usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv",  nrows=1000, usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
print("loading data done!")


# =============================================================================
# Add image quality: by steeve
# ============================================================================= 
import pickle
with open('../input/inception_v3_include_head_max_train.p','rb') as f:
    x = pickle.load(f)
    
train_features = x['features']
train_ids = x['ids']

with open('../input/inception_v3_include_head_max_test.p','rb') as f:
    x = pickle.load(f)

test_features = x['features']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_features, columns = ['image_quality'])
incep_test_image_df = pd.DataFrame(test_features, columns = [f'image_quality'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)

train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')   

del incep_train_image_df, incep_test_image_df
gc.collect()

 
with open('../input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_blurinesses = x['blurinesses']
train_ids = x['ids']

with open('../input/test_image_features.p','rb') as f:
    x = pickle.load(f)

test_blurinesses = x['blurinesses']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_blurinesses, columns = ['blurinesses'])
incep_test_image_df = pd.DataFrame(test_blurinesses, columns = [f'blurinesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding whitenesses ...')
with open('../input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_whitenesses = x['whitenesses']
train_ids = x['ids']


with open('../input/test_image_features.p','rb') as f:
    x = pickle.load(f)

test_whitenesses = x['whitenesses']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_whitenesses, columns = ['whitenesses'])
incep_test_image_df = pd.DataFrame(test_whitenesses, columns = [f'whitenesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding dullnesses ...')
with open('../input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_dullnesses = x['dullnesses']
train_ids = x['ids']

with open('../input/test_image_features.p','rb') as f:
    x = pickle.load(f)

test_dullnesses = x['dullnesses']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_dullnesses, columns = ['dullnesses'])
incep_test_image_df = pd.DataFrame(test_dullnesses, columns = [f'dullnesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


# =============================================================================
# new image data
# =============================================================================

print('adding average_pixel_width ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_average_pixel_width = x['average_pixel_width']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_average_pixel_width = x['average_pixel_width']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_average_pixel_width, columns = ['average_pixel_width'])
incep_test_image_df = pd.DataFrame(test_average_pixel_width, columns = [f'average_pixel_width'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding average_reds ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_average_reds = x['average_reds']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_average_reds = x['average_reds']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_average_reds, columns = ['average_reds'])
incep_test_image_df = pd.DataFrame(test_average_reds, columns = [f'average_reds'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding average_blues ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_average_blues = x['average_blues']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_average_blues = x['average_blues']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_average_blues, columns = ['average_blues'])
incep_test_image_df = pd.DataFrame(test_average_blues, columns = [f'average_blues'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')



print('adding average_greens ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_average_greens = x['average_greens']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_average_greens = x['average_greens']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_average_greens, columns = ['average_greens'])
incep_test_image_df = pd.DataFrame(test_average_greens, columns = [f'average_greens'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding widths ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_widths = x['widths']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_widths = x['widths']
test_ids = x['ids']    
del x; gc.collect()


incep_train_image_df = pd.DataFrame(train_widths, columns = ['widths'])
incep_test_image_df = pd.DataFrame(test_widths, columns = [f'widths'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding heights ...')
with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_heights = x['heights']
train_ids = x['ids']

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_heights = x['heights']
test_ids = x['ids']    
del x; gc.collect()

incep_train_image_df = pd.DataFrame(train_heights, columns = ['heights'])
incep_test_image_df = pd.DataFrame(test_heights, columns = [f'heights'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


del test_average_blues, test_average_greens, test_average_reds, incep_test_image_df
del train_average_blues, train_average_greens, train_average_reds, incep_train_image_df
gc.collect()


#==============================================================================
# image features by Qifeng
#==============================================================================
print('adding image features @ qifeng ...')
with open('../input/train_image_features_cspace.p','rb') as f:
    x = pickle.load(f)

x_train = pd.DataFrame(x, columns = ['average_HSV_Ss',\
                                     'average_HSV_Vs',\
                                     'average_LUV_Ls',\
                                     'average_LUV_Us',\
                                     'average_LUV_Vs',\
                                     'average_HLS_Hs',\
                                     'average_HLS_Ls',\
                                     'average_HLS_Ss',\
                                     'average_YUV_Ys',\
                                     'average_YUV_Us',\
                                     'average_YUV_Vs',\
                                     'ids'
                                     ])
#x_train.rename(columns = {'$ids':'image'}, inplace = True)

with open('../input/test_image_features_cspace.p','rb') as f:
    x = pickle.load(f)

x_test = pd.DataFrame(x, columns = ['average_HSV_Ss',\
                                     'average_HSV_Vs',\
                                     'average_LUV_Ls',\
                                     'average_LUV_Us',\
                                     'average_LUV_Vs',\
                                     'average_HLS_Hs',\
                                     'average_HLS_Ls',\
                                     'average_HLS_Ss',\
                                     'average_YUV_Ys',\
                                     'average_YUV_Us',\
                                     'average_YUV_Vs',\
                                     'ids'
                                    ])
#x_test.rename(columns = {'$ids':'image'}, inplace = True)

train_df = train_df.join(x_train.set_index('ids'), on='image')
test_df = test_df.join(x_test.set_index('ids'), on='image')
del x, x_train, x_test; gc.collect()

# =============================================================================
# add geo info: https://www.kaggle.com/frankherfert/avito-russian-region-cities/data
# =============================================================================
#tmp = pd.read_csv("../input/avito_region_city_features.csv", usecols=["region", "city", "latitude","longitude"])
#train_df = train_df.merge(tmp, on=["city","region"], how="left")
#train_df["lat_long"] = train_df["latitude"]+train_df["longitude"]
#test_df = test_df.merge(tmp, on=["city","region"], how="left")
#test_df["lat_long"] = test_df["latitude"]+test_df["longitude"]
#del tmp; gc.collect()

# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/region_income.csv", sep=";", names=["region", "income"])
train_df = train_df.merge(tmp, on="region", how="left")
test_df = test_df.merge(tmp, on="region", how="left")
del tmp; gc.collect()
# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/city_population_wiki_v3.csv")
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
        
    def Do_Label_Enc(df):
        print("feature engineering -> label encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               "param_1", "param_2", "param_3","image",
               ]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()
    
    import string
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])         
    def Do_NA(df):
        print("feature engineering -> fill na ...")
                
        df["image_top_1"].fillna(-1,inplace=True)
        df["image"].fillna("noinformation",inplace=True)
        df["param_1"].fillna("nicapotato",inplace=True)
        df["param_2"].fillna("nicapotato",inplace=True)
        df["param_3"].fillna("nicapotato",inplace=True)
        df["title"].fillna("nicapotato",inplace=True)
        df["description"].fillna("nicapotato",inplace=True)
        # price vs income
#        df["price_vs_city_income"] = df["price"] / df["income"]
#        df["price_vs_city_income"].fillna(-1, inplace=True)
        
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
        
        # new count 0604
        df["num_title_Exclamation"] = df["title"].apply(lambda x: count(x, "!")).astype(np.int16)
        df["num_title_Question"] = df["title"].apply(lambda x: count(x, "?")).astype(np.int16)
         
        df["num_desc_Exclamation"] = df["description"].apply(lambda x: count(x, "!")).astype(np.int16)
        df["num_desc_Question"] = df["description"].apply(lambda x: count(x, "?")).astype(np.int16)        
                              
    def Do_Drop(df):
        df.drop(["activation_date"], axis=1, inplace=True)
        
    def Do_Stat_Text(df):
        print("feature engineering -> statistics in text ...")
        textfeats = ["text_feature","text_feature_2","description", "title"]
        for col in textfeats:
            df[col + "_num_chars"] = df[col].apply(len).astype(np.int16)
            df[col + "_num_words"] = df[col].apply(lambda comment: len(comment.split())).astype(np.int16)
            df[col + "_num_unique_words"] = df[col].apply(lambda comment: len(set(w for w in comment.split()))).astype(np.int16)
            df[col + "_words_vs_unique"] = (df[col+"_num_unique_words"] / df[col+"_num_words"] * 100).astype(np.float32)
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

#    tfidf_para2 = {
#        "stop_words": russian_stop,
#        "analyzer": "char",
#        "token_pattern": r"\w{1,}",
#        "sublinear_tf": True,
#        "dtype": np.float32,
#        "norm": "l2",
#        # "min_df":5,
#        # "max_df":.9,
#        "smooth_idf": False
#    }

    def get_col(col_name): return lambda x: x[col_name]
    vectorizer = FeatureUnion([
        ("description", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,#40000,18000
            **tfidf_para,
            preprocessor=get_col("description"))
         ),
#         ("title_description", TfidfVectorizer(
#              ngram_range=(1, 2),#(1,2)
#              max_features=1800,#40000,18000
#              **tfidf_para,
#              preprocessor=get_col("title_description"))
#           ), 
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

#        ("title_char", TfidfVectorizer(
#
#            ngram_range=(1, 4),#(1, 4),(1,6)
#            max_features=16000,#16000
#            **tfidf_para2,
#            preprocessor=get_col("title"))
#         ),
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


def get_oof(clf, x_train, y, x_test):
            
    oof_train = np.zeros((len_train,))
    oof_test = np.zeros((len_test,))
    oof_test_skf = np.empty((NFOLDS, len_test))

    for i, (train_index, test_index) in enumerate(kf):
#        print('Ridege oof Fold {}'.format(i))
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

# =============================================================================
# handle price
# =============================================================================
def feature_Eng_On_Price_Make_More_Cat(df):
    print('feature engineering -> on price and SEQ ...')    
    df["price"] = np.log(df["price"]+0.001).astype("float32") 
    df["price"].fillna(-1,inplace=True) 
    df["price+"] = np.round(df["price"]*2.8).astype(np.int16) # 4.8
    df["item_seq_number+"] = np.round(df["item_seq_number"]/100).astype(np.int16)
    
    # by steeve
    df['des_len_log'] = (np.log(df['description_num_chars']) * 4).astype(np.int8)
    df['des_nwords_log'] = (np.log1p(df['description_num_words']) * 20).astype(np.int8)
    return df
  
def feature_Eng_On_Deal_Prob(df, df_train):
    print('feature engineering -> on price deal prob +...')
    df2 = df
    
    # [465]   train's rmse: 0.161946  valid's rmse: 0.22738
    tmp = df_train.groupby(["price+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_price+'})     
    df = pd.merge(df, tmp, how='left', on=["price+"])
    df2['median_deal_probability_price+'] = df['median_deal_probability_price+']
    df2['median_deal_probability_price+'] =df2['median_deal_probability_price+'].astype(np.float32)
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["param_2"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_param_2'})     
    df = pd.merge(df, tmp, how='left', on=["param_2"])
    df2['median_deal_probability_param_2'] = df['median_deal_probability_param_2']
    df2['median_deal_probability_param_2'] =df2['median_deal_probability_param_2'].astype(np.float32)
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["item_seq_number+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_item_seq_number+'})     
    df = pd.merge(df, tmp, how='left', on=["item_seq_number+"])
    df2['median_deal_probability_item_seq_number+'] = df['median_deal_probability_item_seq_number+']
    df2['median_deal_probability_item_seq_number+'] =df2['median_deal_probability_item_seq_number+'].astype(np.float32)
    del tmp; gc.collect()      
    return df2

del train_df, test_df; gc.collect()


# =============================================================================
# use additianl image data
# =============================================================================
feature_engineering(full_df)

# 内存优化
full_df["average_blues"] = full_df["average_blues"].astype(np.float32)
full_df["average_greens"] = full_df["average_greens"].astype(np.float32)
full_df["average_pixel_width"] = full_df["average_pixel_width"].astype(np.float32)
full_df["average_reds"] = full_df["average_reds"].astype(np.float32)
full_df["avg_days_up_user"] = full_df["avg_days_up_user"].astype(np.float32)
full_df["avg_times_up_user"] = full_df["avg_times_up_user"].astype(np.float32)
full_df["blurinesses"] = full_df["blurinesses"].astype(np.float32)
full_df["dullnesses"] = full_df["dullnesses"].astype(np.float32)
full_df["heights"] = full_df["heights"].astype(np.float32)
full_df["parent_category_name"] = full_df["parent_category_name"].astype(np.float32)
full_df["whitenesses"] = full_df["whitenesses"].astype(np.float32)
full_df["widths"] = full_df["widths"].astype(np.float32)
full_df["category_name"] = full_df["category_name"].astype(np.int32)
full_df["city"] = full_df["city"].astype(np.int32)
full_df["image"] = full_df["image"].astype(np.int32)
full_df["image_top_1"] = full_df["image_top_1"].astype(np.int32)
full_df["income"] = full_df["income"].astype(np.int32)
full_df["item_seq_number"] = full_df["item_seq_number"].astype(np.int32)
full_df["n_user_items"] = full_df["n_user_items"].astype(np.int32)
full_df["param_1"] = full_df["param_1"].astype(np.int32)
full_df["param_2"] = full_df["param_2"].astype(np.int32)
full_df["param_3"] = full_df["param_3"].astype(np.int32)
full_df["parent_category_name"] = full_df["parent_category_name"].astype(np.int32)
full_df["region"] = full_df["region"].astype(np.int32)
full_df["user_id"] = full_df["user_id"].astype(np.int32)
full_df["user_type"] = full_df["user_type"].astype(np.int32)
full_df["population"] = full_df["population"].fillna(-1).astype(np.int32)
full_df["average_HLS_Hs"] = full_df["average_HLS_Hs"].astype(np.float32)
full_df["average_HLS_Ls"] = full_df["average_HLS_Ls"].astype(np.float32)
full_df["average_HLS_Ss"] = full_df["average_HLS_Ss"].astype(np.float32)
full_df["average_HSV_Ss"] = full_df["average_HSV_Ss"].astype(np.float32)
full_df["average_HSV_Vs"] = full_df["average_HSV_Vs"].astype(np.float32)
full_df["average_LUV_Ls"] = full_df["average_LUV_Ls"].astype(np.float32)
full_df["average_LUV_Us"] = full_df["average_LUV_Us"].astype(np.float32)
full_df["average_LUV_Vs"] = full_df["average_LUV_Vs"].astype(np.float32)
full_df["average_YUV_Us"] = full_df["average_YUV_Us"].astype(np.float32)
full_df["average_YUV_Vs"] = full_df["average_YUV_Vs"].astype(np.float32)
full_df["average_YUV_Ys"] = full_df["average_YUV_Ys"].astype(np.float32)

gc.collect()


from sklearn.model_selection import KFold
kf2 = KFold(n_splits=10, random_state=42, shuffle=True)
numIter = 0
rmse_sume = 0.
numLimit = 2

tmp = pd.DataFrame(full_df)
full_df_COPY = pd.DataFrame(tmp)
del tmp

for train_index, valid_index in kf2.split(y):
      numIter +=1
      print("training in fold " + str(numIter))
      
      if numIter>=numLimit+1:
            pass
      else:       
            full_df = pd.DataFrame(full_df_COPY)
            tmp = full_df[:len_train]
            train_df = tmp.iloc[train_index]
            del tmp;gc.collect()
                        
            # 不考虑使用均值
            try:
                  full_df.drop('median_deal_probability_price+', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_price+', axis=1, inplace=True); gc.collect()
                  full_df.drop('median_deal_probability_param_2', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_param_2', axis=1, inplace=True); gc.collect()
                  full_df.drop('median_deal_probability_item_seq_number+', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_item_seq_number+', axis=1, inplace=True); gc.collect()
            except:
                  pass                  
            
            feature_Eng_On_Price_Make_More_Cat(full_df)
            feature_Eng_On_Price_Make_More_Cat(train_df)
            feature_Eng_On_Deal_Prob(full_df, train_df)
                        
            try:
                  full_df.drop('deal_probability', axis=1, inplace=True); gc.collect()
            except:
                  pass
            
            full_df, ready_full_df, tfvocab = data_vectorize(full_df)
            ready_df = ready_full_df
                        
            # NN
#            print("load nn oof 1 ...")
#            nn_oof_train = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_train.csv')
#            nn_oof_train.drop("user_id", axis=1, inplace=True)
#            nn_oof_test = pd.read_csv('../input/emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_5fold_test.csv')
#            
#            nn_oof_full = pd.concat([nn_oof_train, nn_oof_test])
#            nn_oof_full["nn_oof"] = nn_oof_full["deal_probability"]
#            del nn_oof_full["deal_probability"]
#            full_df = pd.merge(full_df, nn_oof_full, on='item_id', how='left')
#            del nn_oof_train, nn_oof_test
#            gc.collect()
                                   
            from sklearn.cross_validation import KFold
            NFOLDS = 10#5
            SEED = 42
            kf = KFold(len_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)
            
            # SGD
            from sklearn.linear_model import SGDRegressor
            sgdregressor_params = {'alpha':0.0001, 'random_state':SEED, 'tol':1e-3}
            sgd = SklearnWrapper(clf=SGDRegressor, seed = SEED, params = sgdregressor_params)
            FULL_DF = pd.DataFrame(full_df)
            FULL_DF.drop(["item_id"], axis=1, inplace=True)
            tmp1 = pd.DataFrame(full_df)
            tmp1.drop(["item_id"], axis=1, inplace=True)
            print('sgd 1 oof ...')
            sgd_oof_train, sgd_oof_test = get_oof(sgd, np.array(FULL_DF)[:len_train], y, np.array(FULL_DF)[len_train:])
            sgd_preds = np.concatenate([sgd_oof_train, sgd_oof_test])
            tmp1['sgd_preds_1'] = sgd_preds.astype(np.float32)
            tmp1['sgd_preds_1'].clip(0.0, 1.0, inplace=True)
            print('sgd 2 oof ...')
            sgd_oof_train, sgd_oof_test = get_oof(sgd, ready_df[:len_train], y, ready_df[len_train:])
            sgd_preds = np.concatenate([sgd_oof_train, sgd_oof_test])
            tmp1['sgd_preds_2'] = sgd_preds.astype(np.float32)
            tmp1['sgd_preds_2'].clip(0.0, 1.0, inplace=True)

            # Ridge
            #'alpha':20.0
            ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                            'max_iter':None, 'tol':1e-3, 'solver':'auto', 'random_state':SEED}
            ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
            FULL_DF = pd.DataFrame(full_df)
            FULL_DF.drop(["item_id"], axis=1, inplace=True)
            tmp2 = pd.DataFrame(full_df)
            tmp2.drop(["item_id"], axis=1, inplace=True)
            print('ridge 1 oof ...')
            ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(FULL_DF)[:len_train], y, np.array(FULL_DF)[len_train:])
            ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
            tmp2['ridge_preds_1'] = ridge_preds.astype(np.float32)
            tmp2['ridge_preds_1'].clip(0.0, 1.0, inplace=True)
            print('ridge 2 oof ...')
            ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:])
            ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
            tmp2['ridge_preds_2'] = ridge_preds.astype(np.float32)
            tmp2['ridge_preds_2'].clip(0.0, 1.0, inplace=True)

            ## Ridge
            ##'alpha':20.0
            ridge_params = {'alpha':10.0, 'fit_intercept':True, 'normalize':True, 'copy_X':True,
                            'max_iter':None, 'tol':1e-3, 'solver':'auto', 'random_state':SEED+2011}
            ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
            FULL_DF = pd.DataFrame(full_df)
            FULL_DF.drop(["item_id"], axis=1, inplace=True)
            tmp3 = pd.DataFrame(full_df)
            tmp3.drop(["item_id"], axis=1, inplace=True)
            print('ridge 1a oof ...')
            ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(FULL_DF)[:len_train], y, np.array(FULL_DF)[len_train:])
            ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
            tmp3['ridge_preds_1a'] = ridge_preds.astype(np.float32)
            tmp3['ridge_preds_1a'].clip(0.0, 1.0, inplace=True)
            print('ridge 2a oof ...')
            ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:])
            ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
            tmp3['ridge_preds_2a'] = ridge_preds.astype(np.float32)
            tmp3['ridge_preds_2a'].clip(0.0, 1.0, inplace=True)
                        
            # 融入oof结果
            full_df['sgd_preds_1'] = tmp1['sgd_preds_1'].astype(np.float32)
            full_df['sgd_preds_2'] = tmp1['sgd_preds_2'].astype(np.float32)
            
            full_df['ridge_preds_1'] = tmp2['ridge_preds_1'].astype(np.float32)
            full_df['ridge_preds_2'] = tmp2['ridge_preds_2'].astype(np.float32)
            
            full_df['ridge_preds_1a'] = tmp3['ridge_preds_1a'].astype(np.float32)
            full_df['ridge_preds_2a'] = tmp3['ridge_preds_2a'].astype(np.float32)
            
            del tmp1, tmp2, tmp3
            del ridge_oof_train, ridge_oof_test, ridge_preds, ridge, sgd_oof_test, sgd_oof_train, sgd_preds, ready_df
            gc.collect()
                        

                        
            full_df.drop("item_id", axis=1, inplace=True)
            
            print("Modeling Stage ...")
            # Combine Dense Features with Sparse Text Bag of Words Features
            X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
            tfvocab = full_df.columns.tolist() + tfvocab
            X_test_full=full_df.iloc[len_train:]
            X_test_ready=ready_full_df[len_train:]
            del ready_full_df, full_df
            gc.collect()
                        
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
#                       "des_len_log",
#                       "des_nwords_log",
                       ]
                
            print("Modeling Stage ...")             
            X_train, X_valid = X.tocsr()[train_index], X.tocsr()[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]            
            gc.collect()            
            lgbm_params =  {
#                    "gpu_platform_id":-1,
#                    "gpu_device_id":-1,
                    "tree_learner": "feature",    
                    "num_threads": 3,
                    "task": "train",
                    "boosting_type": "gbdt",
                    "objective": "regression",
                    "metric": "rmse",
                    "num_leaves": 500, # 500, 280,360,500,32
                    "feature_fraction": 0.2, #0.4
                    "bagging_fraction": 0.2, #0.4
                    "learning_rate": 0.015,#0.015
                    "verbose": -1,
                    'lambda_l1':1,
                    'lambda_l2':1,
                    "max_bin":200,
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
                  
            del X_train, X_valid, y_train, y_valid, lgtrain, lgvalid, train_df
            gc.collect()
      

print("mean rmse is:", rmse_sume/numLimit)      
print("Features importance...")
bst = lgb_clf
gain = bst.feature_importance("gain")
ft = pd.DataFrame({"feature":bst.feature_name(), "split":bst.feature_importance("split"), "gain":100 * gain / gain.sum()}).sort_values("gain", ascending=False)
print(ft.head(50))
#
#plt.figure()
#ft[["feature","gain"]].head(50).plot(kind="barh", x="feature", y="gain", legend=False, figsize=(10, 20))
#plt.gcf().savefig("features_importance.png")

print("All Done.")


"""
(test)
[100]   train's rmse: 0.224476  valid's rmse: 0.226889
[200]   train's rmse: 0.216223  valid's rmse: 0.221169
[300]   train's rmse: 0.211851  valid's rmse: 0.21918
[400]   train's rmse: 0.20819   valid's rmse: 0.2179
[500]   train's rmse: 0.205255  valid's rmse: 0.217123
[600]   train's rmse: 0.202585  valid's rmse: 0.216582
[700]   train's rmse: 0.200149  valid's rmse: 0.216154
[800]   train's rmse: 0.197939  valid's rmse: 0.215845
[900]   train's rmse: 0.195978  valid's rmse: 0.215654
[1000]  train's rmse: 0.19414   valid's rmse: 0.215491



[100]   train's rmse: 0.225319  valid's rmse: 0.227587
[200]   train's rmse: 0.216781  valid's rmse: 0.221473
[300]   train's rmse: 0.212228  valid's rmse: 0.219283
[400]   train's rmse: 0.208614  valid's rmse: 0.218002
[500]   train's rmse: 0.205568  valid's rmse: 0.217158
[600]   train's rmse: 0.203035  valid's rmse: 0.216592
[700]   train's rmse: 0.200543  valid's rmse: 0.216153
[800]   train's rmse: 0.198402  valid's rmse: 0.215871
[900]   train's rmse: 0.196333  valid's rmse: 0.215609
[1000]  train's rmse: 0.194439  valid's rmse: 0.215434
[1100]  train's rmse: 0.1926    valid's rmse: 0.215294
[1200]  train's rmse: 0.190889  valid's rmse: 0.215184
[1300]  train's rmse: 0.189277  valid's rmse: 0.215118
[1400]  train's rmse: 0.187809  valid's rmse: 0.215061
[1500]  train's rmse: 0.186371  valid's rmse: 0.215005
[1600]  train's rmse: 0.184991  valid's rmse: 0.214946
[1700]  train's rmse: 0.183598  valid's rmse: 0.214905
[1800]  train's rmse: 0.182244  valid's rmse: 0.214872
[1900]  train's rmse: 0.180993  valid's rmse: 0.214845
[2000]  train's rmse: 0.179789  valid's rmse: 0.214832
[2100]  train's rmse: 0.178642  valid's rmse: 0.214833
[2200]  train's rmse: 0.177541  valid's rmse: 0.214813
[2300]  train's rmse: 0.176456  valid's rmse: 0.214779
[2400]  train's rmse: 0.175414  valid's rmse: 0.214761
[2500]  train's rmse: 0.174438  valid's rmse: 0.214738
[2600]  train's rmse: 0.173494  valid's rmse: 0.214728
[2700]  train's rmse: 0.17259   valid's rmse: 0.214719
[2800]  train's rmse: 0.171668  valid's rmse: 0.214718
[2900]  train's rmse: 0.170743  valid's rmse: 0.214714
[3000]  train's rmse: 0.16985   valid's rmse: 0.214707
[3100]  train's rmse: 0.169026  valid's rmse: 0.214701
[3200]  train's rmse: 0.168146  valid's rmse: 0.214684
[3300]  train's rmse: 0.167339  valid's rmse: 0.214677
[3400]  train's rmse: 0.166494  valid's rmse: 0.214685
Early stopping, best iteration is:
[3244]  train's rmse: 0.167777  valid's rmse: 0.214673
save model ...
Model Evaluation Stage
calculating RMSE ...
RMSE: 0.21467256607424468

calculating RMSE ...
training in fold 2
Modeling Stage ...
Training until validation scores don't improve for 200 rounds.
[100]   train's rmse: 0.224584  valid's rmse: 0.226573
[200]   train's rmse: 0.21663   valid's rmse: 0.221173
[300]   train's rmse: 0.212054  valid's rmse: 0.218924
[400]   train's rmse: 0.208659  valid's rmse: 0.21754
[500]   train's rmse: 0.205683  valid's rmse: 0.216717
[600]   train's rmse: 0.203098  valid's rmse: 0.216138
[700]   train's rmse: 0.20075   valid's rmse: 0.21565
[800]   train's rmse: 0.198486  valid's rmse: 0.215285
[900]   train's rmse: 0.196409  valid's rmse: 0.215011
[1000]  train's rmse: 0.194553  valid's rmse: 0.214818
[1100]  train's rmse: 0.192716  valid's rmse: 0.214652
[1200]  train's rmse: 0.190952  valid's rmse: 0.214545
[1300]  train's rmse: 0.189312  valid's rmse: 0.214432
[1400]  train's rmse: 0.187826  valid's rmse: 0.214345
[1500]  train's rmse: 0.186349  valid's rmse: 0.214291
[1600]  train's rmse: 0.184924  valid's rmse: 0.214254
[1700]  train's rmse: 0.183661  valid's rmse: 0.214223
[1800]  train's rmse: 0.182386  valid's rmse: 0.214203
[1900]  train's rmse: 0.181181  valid's rmse: 0.214174
[2000]  train's rmse: 0.179975  valid's rmse: 0.214142
[2100]  train's rmse: 0.178874  valid's rmse: 0.21412
[2200]  train's rmse: 0.177713  valid's rmse: 0.214107
[2300]  train's rmse: 0.17663   valid's rmse: 0.214091
[2400]  train's rmse: 0.175612  valid's rmse: 0.214072
[2500]  train's rmse: 0.174632  valid's rmse: 0.214072
[2600]  train's rmse: 0.173608  valid's rmse: 0.214065
[2700]  train's rmse: 0.172698  valid's rmse: 0.214062
[2800]  train's rmse: 0.171783  valid's rmse: 0.214062
[2900]  train's rmse: 0.170862  valid's rmse: 0.21406
[3000]  train's rmse: 0.170011  valid's rmse: 0.214053
[3100]  train's rmse: 0.169159  valid's rmse: 0.214059
[3200]  train's rmse: 0.168311  valid's rmse: 0.214062
Early stopping, best iteration is:
[3029]  train's rmse: 0.169765  valid's rmse: 0.214051
save model ...
Model Evaluation Stage
calculating RMSE ...
RMSE: 0.21405066855179888
calculating RMSE ...
training in fold 3
training in fold 4
training in fold 5
training in fold 6
training in fold 7
training in fold 8
training in fold 9
training in fold 10
mean rmse is: 0.21436161731302178
Features importance...
                               feature       gain  split
13                         image_top_1  11.936618  89246
68                       ridge_preds_2   8.394325  20879
8                                 city   6.168863  76198
17                             param_1   6.156261  32437
7                        category_name   5.601058  18024
67                       ridge_preds_1   3.376310  20287
18                             param_2   3.195307  17448
20                parent_category_name   2.121094   4217
65                              price+   1.964126  20488
16                        n_user_items   1.884413  14190
19                             param_3   1.866375  17733
22                               price   1.756159  19376
23                              region   1.593146  35508
4                     avg_days_up_user   1.111384  15647
24                             user_id   1.002529  12751
15                     item_seq_number   0.764631  18375
5                    avg_times_up_user   0.643270  11443
0                        average_blues   0.591781  16710
66                    item_seq_number+   0.585969   6494
3                         average_reds   0.581833  16502
1                       average_greens   0.558866  15964
21                          population   0.546603  14724
57               description_num_chars   0.473153  14454
6                          blurinesses   0.468096  16982
2                  average_pixel_width   0.445799  16228
43651             text_feature__одежда   0.435206    370
58               description_num_words   0.429626  11371
49              text_feature_num_chars   0.412445   6619
12                       image_quality   0.400337  16549
11                               image   0.383210  16071
27                              widths   0.369020   8344
55     text_feature_2_num_unique_words   0.350646   1384
25                           user_type   0.350333   2221
59        description_num_unique_words   0.342265  10217
28                      num_desc_punct   0.341784   9153
61                     title_num_chars   0.336563  11496
54            text_feature_2_num_words   0.311537   1070
33339           description__состоянии   0.299048   5860
9                           dullnesses   0.249431   8615
60         description_words_vs_unique   0.242027   8732
26                         whitenesses   0.240906   7796
14                              income   0.240634   9397
27954              description__продам   0.206935   5148
41576         text_feature__nicapotato   0.200321    729
51       text_feature_num_unique_words   0.186407   2467
53            text_feature_2_num_chars   0.177795   2679
50              text_feature_num_words   0.165370   2258
52        text_feature_words_vs_unique   0.163689   2588
43738          text_feature__перевозки   0.160041    159
62                     title_num_words   0.159509   4947
All Done.

correlation between models outputs
         dp1      dp2
dp1  1.00000  0.98581
dp2  0.98581  1.00000

"""
