from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack, csr_matrix
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import gc, re
from sklearn.utils import shuffle
from contextlib import contextmanager
#from sklearn.externals import joblib
from sklearn import preprocessing
import time

# =============================================================================
# Keras
# =============================================================================
import pyximport
pyximport.install()
import os
import random
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '10000'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(10001)
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend
tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
from keras.layers import Input, Dropout, Dense, concatenate, PReLU,SpatialDropout1D, GaussianDropout, Embedding, Flatten, Activation, BatchNormalization
from keras.initializers import he_uniform, RandomNormal
#from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
#from keras import regularizers


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
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:100000]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv",  nrows=1000, parse_dates = ["activation_date"])
    
    train_active = pd.read_csv("../input/train_active.csv",  nrows=1000, usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv",  nrows=1000, usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
print("loading data done!")

label = ['deal_probability']
train_user_ids = train_df.user_id.values
train_item_ids = train_df.item_id.values

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
incep_test_image_df = pd.DataFrame(test_features, columns = ['image_quality'])
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
incep_test_image_df = pd.DataFrame(test_blurinesses, columns = ['blurinesses'])
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
incep_test_image_df = pd.DataFrame(test_whitenesses, columns = ['whitenesses'])
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
incep_test_image_df = pd.DataFrame(test_dullnesses, columns = ['dullnesses'])
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
incep_test_image_df = pd.DataFrame(test_average_pixel_width, columns = ['average_pixel_width'])
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
incep_test_image_df = pd.DataFrame(test_average_reds, columns = ['average_reds'])
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
incep_test_image_df = pd.DataFrame(test_average_blues, columns = ['average_blues'])
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
incep_test_image_df = pd.DataFrame(test_average_greens, columns = ['average_greens'])
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
incep_test_image_df = pd.DataFrame(test_widths, columns = ['widths'])
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
incep_test_image_df = pd.DataFrame(test_heights, columns = ['heights'])
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
    df["price_p"] = np.round(df["price"]*2.8).astype(np.int16) # 4.8
    df["item_seq_number_p"] = np.round(df["item_seq_number"]/100).astype(np.int16)
    df["price"] = np.expm1(df["price"]).astype("float32")
    return df
  
def feature_Eng_On_Deal_Prob(df, df_train):
    print('feature engineering -> on price deal prob +...')
    df2 = df
    
    # [465]   train's rmse: 0.161946  valid's rmse: 0.22738
    tmp = df_train.groupby(["price_p"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_price_p'})     
    df = pd.merge(df, tmp, how='left', on=["price_p"])
    df2['median_deal_probability_price_p'] = df['median_deal_probability_price_p']
    df2['median_deal_probability_price_p'] =df2['median_deal_probability_price_p'].astype(np.float32)
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["param_2"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_param_2'})     
    df = pd.merge(df, tmp, how='left', on=["param_2"])
    df2['median_deal_probability_param_2'] = df['median_deal_probability_param_2']
    df2['median_deal_probability_param_2'] =df2['median_deal_probability_param_2'].astype(np.float32)
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["item_seq_number_p"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_item_seq_number_p'})     
    df = pd.merge(df, tmp, how='left', on=["item_seq_number_p"])
    df2['median_deal_probability_item_seq_number_p'] = df['median_deal_probability_item_seq_number_p']
    df2['median_deal_probability_item_seq_number_p'] =df2['median_deal_probability_item_seq_number_p'].astype(np.float32)
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
full_df["parent_category_name"] = full_df["parent_category_name"].astype(np.int32)
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
full_df.drop("image", axis=1, inplace=True)
full_df.drop("user_id", axis=1, inplace=True)
gc.collect()


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model(X_train):                                             
    sparse_data = Input( shape=[X_train["sparse_data"].shape[1]], 
        dtype = 'float32',   sparse = True, name='sparse_data')  

    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    parent_category_name = Input(shape=[1], name="parent_category_name")                
    category_name = Input(shape=[1], name="category_name")
    user_type = Input(shape=[1], name="user_type")                
    image_top_1 = Input(shape=[1], name="image_top_1")                
    param_1 = Input(shape=[1], name="param_1")                
    param_2 = Input(shape=[1], name="param_2")                
    param_3 = Input(shape=[1], name="param_3")                
    price_p = Input(shape=[1], name="price_p")
    item_seq_number_p = Input(shape=[1], name="item_seq_number_p") 
    
    median_deal_probability_price_p = Input(shape=[1], name="median_deal_probability_price_p") 
    median_deal_probability_param_2 = Input(shape=[1], name="median_deal_probability_param_2") 
    median_deal_probability_item_seq_number_p = Input(shape=[1], name="median_deal_probability_item_seq_number_p")
    
    sgd_preds_1 = Input(shape=[1], name="sgd_preds_1") 
    sgd_preds_2 = Input(shape=[1], name="sgd_preds_2") 
    ridge_preds_1 = Input(shape=[1], name="ridge_preds_1") 
    ridge_preds_2 = Input(shape=[1], name="ridge_preds_2") 
    ridge_preds_1a = Input(shape=[1], name="ridge_preds_1a") 
    ridge_preds_2a = Input(shape=[1], name="ridge_preds_2a") 
    ridge_preds_3 = Input(shape=[1], name="ridge_preds_3")
    
    average_HLS_Hs = Input(shape=[1], name="average_HLS_Hs") 
    average_HLS_Ls = Input(shape=[1], name="average_HLS_Ls") 
    average_HLS_Ss = Input(shape=[1], name="average_HLS_Ss") 
    average_HSV_Ss = Input(shape=[1], name="average_HSV_Ss") 
    average_HSV_Vs = Input(shape=[1], name="average_HSV_Vs") 
    average_LUV_Ls = Input(shape=[1], name="average_LUV_Ls") 
    average_LUV_Us = Input(shape=[1], name="average_LUV_Us") 
    average_LUV_Vs = Input(shape=[1], name="average_LUV_Vs") 
    average_YUV_Us = Input(shape=[1], name="average_YUV_Us") 
    average_YUV_Vs = Input(shape=[1], name="average_YUV_Vs") 
    average_YUV_Ys = Input(shape=[1], name="average_YUV_Ys")
    average_reds = Input(shape=[1], name="average_reds") 
    average_blues = Input(shape=[1], name="average_blues") 
    average_greens = Input(shape=[1], name="average_greens")
    average_pixel_width = Input(shape=[1], name="average_pixel_width") 
    blurinesses = Input(shape=[1], name="blurinesses") 
    dullnesses = Input(shape=[1], name="dullnesses") 
    heights = Input(shape=[1], name="heights") 
    image_quality = Input(shape=[1], name="image_quality") 
    
    
    text_feature_num_chars = Input(shape=[1], name="text_feature_num_chars") 
    text_feature_num_words = Input(shape=[1], name="text_feature_num_words") 
    text_feature_num_unique_words = Input(shape=[1], name="text_feature_num_unique_words") 
    text_feature_words_vs_unique = Input(shape=[1], name="text_feature_words_vs_unique") 
    text_feature_2_num_chars = Input(shape=[1], name="text_feature_2_num_chars") 
    text_feature_2_num_words = Input(shape=[1], name="text_feature_2_num_words") 
    text_feature_2_num_unique_words = Input(shape=[1], name="text_feature_2_num_unique_words") 
    text_feature_2_words_vs_unique = Input(shape=[1], name="text_feature_2_words_vs_unique") 
    description_num_chars = Input(shape=[1], name="description_num_chars") 
    description_num_words = Input(shape=[1], name="description_num_words") 
    description_words_vs_unique = Input(shape=[1], name="description_words_vs_unique")
    title_num_chars = Input(shape=[1], name="title_num_chars") 
    title_num_words = Input(shape=[1], name="title_num_words") 
    title_num_unique_words = Input(shape=[1], name="title_num_unique_words")
    title_words_vs_unique = Input(shape=[1], name="title_words_vs_unique")
    description_num_unique_words = Input(shape=[1], name="description_num_unique_words") 
    
    avg_days_up_user = Input(shape=[1], name="avg_days_up_user") 
    avg_times_up_user = Input(shape=[1], name="avg_times_up_user") 
    income = Input(shape=[1], name="income") 
    n_user_items = Input(shape=[1], name="n_user_items") 
    population = Input(shape=[1], name="population") 
    price = Input(shape=[1], name="price") 
    wday = Input(shape=[1], name="wday") 
    
    
    num_desc_punct = Input(shape=[1], name="num_desc_punct") 
    num_desc_capE = Input(shape=[1], name="num_desc_capE") 
    num_desc_capP = Input(shape=[1], name="num_desc_capP") 
    num_title_punct = Input(shape=[1], name="num_title_punct") 
    num_title_capE = Input(shape=[1], name="num_title_capE") 
    num_title_capP = Input(shape=[1], name="num_title_capP") 
    is_in_desc_1 = Input(shape=[1], name="is_in_desc_1") 
    
    is_in_desc_2 = Input(shape=[1], name="is_in_desc_2") 
    is_in_desc_3 = Input(shape=[1], name="is_in_desc_3") 
    is_in_desc_4 = Input(shape=[1], name="is_in_desc_4") 
    is_in_desc_5 = Input(shape=[1], name="is_in_desc_5") 
    is_in_desc_6 = Input(shape=[1], name="is_in_desc_6") 
    is_in_desc_7 = Input(shape=[1], name="is_in_desc_7") 
    is_in_desc_8 = Input(shape=[1], name="is_in_desc_8") 
    
    is_in_desc_9 = Input(shape=[1], name="is_in_desc_9") 
    is_in_desc_10 = Input(shape=[1], name="is_in_desc_10") 
    num_title_Exclamation = Input(shape=[1], name="num_title_Exclamation") 
    num_title_Question = Input(shape=[1], name="num_title_Question") 
    num_desc_Exclamation = Input(shape=[1], name="num_desc_Exclamation") 
    num_desc_Question = Input(shape=[1], name="num_desc_Question") 


    hyper_params={
    'description_filters':40,
    'embedding_dim':80,
    'enable_deep':False,
    'enable_fm':True,
    "fc_dim": 64,
    'learning_rate':0.0001
    }
    
    print(hyper_params, flush=True)
    
    def gauss_init():
          return RandomNormal(mean=0.0, stddev=0.005)

    
    max_user_type= np.max(full_df['user_type'].max()) + 1
    max_parent_category_name= np.max(full_df['parent_category_name'].max()) + 1
    max_category_name= np.max(full_df['category_name'].max()) + 1
    max_param_1 = np.max(full_df['param_1'].max()) + 1
    max_param_2 = np.max(full_df['param_2'].max()) + 1
    max_param_3 = np.max(full_df['param_3'].max()) + 1
    max_region = np.max(full_df['region'].max()) + 1
    max_city = np.max(full_df['city'].max()) + 1
    max_image_top_1 = np.max(full_df['image_top_1'].max()) + 1
    max_price_p = np.max(full_df['price_p'].max())+1
    max_item_seq_number_p = np.max(full_df['item_seq_number_p'].max()) + 1
    
    # emb layer def
    emb_user_type =  ( Embedding(max_user_type, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(user_type))
    emb_param_1 =  ( Embedding(max_param_1, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_1) )
    emb_param_2 =  ( Embedding(max_param_2, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_2) )
    emb_param_3 = ( Embedding(max_param_3, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_3) )
    emb_parent_category_name = ( Embedding(max_parent_category_name, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(parent_category_name) )
    emb_category_name =   ( Embedding(max_category_name, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(category_name) )
    
    emb_region = ( Embedding(max_region, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(region) )
    emb_city = ( Embedding(max_city, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(city) )
    emb_image_top_1 =  ( Embedding(max_image_top_1, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(image_top_1) )
    emb_price_p = ( Embedding(max_price_p, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(price_p) )
    emb_item_seq_number_p = ( Embedding(max_item_seq_number_p, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(item_seq_number_p) )

  
    # sparse matrix layer
    # mean valid rmse:  0.2503963423301487                            
    x1 = Dense(128, input_dim=256,
              kernel_initializer=he_uniform(seed=0),
#              kernel_regularizer=regularizers.l2(0.001),
#              activity_regularizer=regularizers.l1(0.001)                      
              )(sparse_data)  
    x1 = BatchNormalization()(x1) 
    x1 = GaussianDropout(0.25)(x1)
    x1 = PReLU()(x1)


    # categorical layer
    x2 = concatenate( [
                      Flatten() (emb_region), 
                      Flatten() (emb_city), 
                      Flatten() (emb_category_name), 
                      Flatten() (emb_parent_category_name),
                      Flatten() (emb_user_type), 
                      Flatten() (emb_param_1), 
                      Flatten() (emb_param_2), 
                      Flatten() (emb_param_3), 
                      Flatten() (emb_image_top_1),  
                      Flatten() (emb_price_p),
                      Flatten() (emb_item_seq_number_p),
                      ])
    
    x2 = concatenate( [x2, region, city, parent_category_name, category_name,
                      user_type, image_top_1, param_1, param_2, param_3, price_p, 
                      item_seq_number_p,
                      ] )
      
    x2 = Dropout(0.05)(x2)
    x2 = Dense(128, kernel_initializer=he_uniform(seed=0))(x2)
    x2 = PReLU()(x2)
#    x2 = Dropout(0.05)(x2)
#    x2 = Dense(128, kernel_initializer=he_uniform(seed=0))(x2)
#    x2 = PReLU()(x2)
       
    # numerical layer 
    x3  = concatenate([
                      median_deal_probability_price_p, median_deal_probability_param_2, median_deal_probability_item_seq_number_p,
                      sgd_preds_1, sgd_preds_2, ridge_preds_1, ridge_preds_2, ridge_preds_1a, ridge_preds_2a, ridge_preds_3
                      ])
      
    x3  = concatenate( [x3 ,
                      avg_days_up_user, avg_times_up_user, income,
                      n_user_items, population, price, wday
                      ])
             
    x3 = Dropout(0.05)(x3)        
    x3 = Dense(128 , kernel_initializer=he_uniform(seed=0))(x3)      
    x3 = PReLU()(x3)

      
    # numerical layer (text_count)
    x4 = concatenate( [
                      text_feature_num_chars, text_feature_num_words, text_feature_num_unique_words,
                      text_feature_words_vs_unique, text_feature_2_num_chars, text_feature_2_num_words, 
                      text_feature_2_num_unique_words, text_feature_2_words_vs_unique, description_num_chars, 
                      description_num_words, description_num_unique_words, description_words_vs_unique, 
                      title_num_chars, title_num_words, title_num_unique_words, title_words_vs_unique
                      ])
     
    x4 = concatenate( [x4,
                      num_desc_punct, num_desc_capE, num_desc_capP,
                      num_title_punct, num_title_capE, num_title_capP, 
                      is_in_desc_1, is_in_desc_2, is_in_desc_3, 
                      is_in_desc_4, is_in_desc_5, is_in_desc_6, 
                      is_in_desc_7, is_in_desc_8, is_in_desc_9, 
                      is_in_desc_10, num_title_Exclamation, num_title_Question, 
                      num_desc_Exclamation, num_desc_Question
                      ])

    x4 = Dropout(0.05)(x4)        
    x4 = Dense(128 , kernel_initializer=he_uniform(seed=0))(x4)      
    x4 = PReLU()(x4)
   
     
    # image layer
    x5  = concatenate( [
                      average_HLS_Hs,average_HLS_Ls,average_HLS_Ss, average_HSV_Ss, average_HSV_Vs,
                      average_LUV_Ls, average_LUV_Us, average_LUV_Vs, average_YUV_Us, 
                      average_YUV_Vs, average_YUV_Ys, average_blues, average_greens, average_pixel_width, 
                      blurinesses, dullnesses, heights, image_quality, average_reds
                      ])

    x5 = Dropout(0.05)(x5)        
    x5 = Dense(128 , kernel_initializer=he_uniform(seed=0))(x5)      
    x5 = PReLU()(x5)
    x5 = Dropout(0.05)(x5)        
    x5 = Dense(128 , kernel_initializer=he_uniform(seed=0))(x5)      
    x5 = PReLU()(x5)
    x5 = Dropout(0.05)(x5)        
      
    x = concatenate([x1,
                     x2,
                     x3,
                     x4,
                     x5
                     ])
    
   
    x = Dropout(0.1)(x)      
    x = Dense(128 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
#    
    x = Dropout(0.1)(x)      
    x = Dense(128 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
    
    x = Dropout(0.1)(x)      
    x = Dense(128 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
    
    x = Dropout(0.1)(x)      
    x = Dense(128 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
    
    x = Dropout(0.1)(x)      
    x = Dense(64 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
    
    x = Dropout(0.1)(x)      
    x = Dense(32 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
    
    x = Dropout(0.1)(x)      
    x = Dense(16 , kernel_initializer=he_uniform(seed=0))(x)       
    x = PReLU()(x)
       
    # output layer    
    x = Dense(1)(x)
   
    model = Model([sparse_data, region, city, parent_category_name, category_name,
                      user_type, image_top_1, param_1, param_2, param_3, price_p, 
                      item_seq_number_p,
                      median_deal_probability_price_p, median_deal_probability_param_2, median_deal_probability_item_seq_number_p,
                      sgd_preds_1, sgd_preds_2, ridge_preds_1, ridge_preds_2, ridge_preds_1a, ridge_preds_2a, ridge_preds_3,
                      
                      average_HLS_Hs,average_HLS_Ls,average_HLS_Ss, average_HSV_Ss, average_HSV_Vs,
                      average_LUV_Ls, average_LUV_Us, average_LUV_Vs, average_YUV_Us, 
                      average_YUV_Vs, average_YUV_Ys, average_blues, average_greens, average_pixel_width, 
                      blurinesses, dullnesses, heights, image_quality, average_reds, 
                      
                      text_feature_num_chars, text_feature_num_words, text_feature_num_unique_words,
                      text_feature_words_vs_unique, text_feature_2_num_chars, text_feature_2_num_words, 
                      text_feature_2_num_unique_words, text_feature_2_words_vs_unique, description_num_chars, 
                      description_num_words, description_num_unique_words, description_words_vs_unique, 
                      title_num_chars, title_num_words, title_num_unique_words, title_words_vs_unique,
                                            
                      avg_days_up_user, avg_times_up_user, income,
                      n_user_items, population, price, wday,
                      
                      num_desc_punct, num_desc_capE, num_desc_capP,
                      num_title_punct, num_title_capE, num_title_capP, 
                      is_in_desc_1, is_in_desc_2, is_in_desc_3, 
                      is_in_desc_4, is_in_desc_5, is_in_desc_6, 
                      is_in_desc_7, is_in_desc_8, is_in_desc_9, 
                      is_in_desc_10, num_title_Exclamation, num_title_Question, 
                      num_desc_Exclamation, num_desc_Question                                         
                      ], x)

    # optimizer = Adam(.0011)
    model.compile(loss=root_mean_squared_error, optimizer="rmsprop")
    return model


from sklearn.model_selection import KFold
kf2 = KFold(n_splits=5, random_state=42, shuffle=True)
numIter = 0
rmse_sume = 0.
numLimit = 5
mean_train_rmse = 0.
mean_valid_rmse = 0.
tmp = pd.DataFrame(full_df)
full_df_COPY = pd.DataFrame(tmp)
del tmp; gc.collect()
val_predict = np.zeros(y.shape)

for train_index, valid_index in kf2.split(y):
      
      numIter +=1
      print("training in fold " + str(numIter))
      
      file_path="NN_ML"+str(numIter)+".hdf5"
      print('file name is:', file_path)
      
      
      checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
      early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
      lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.05,
                                     patience=2,
                                     verbose=2,
                                     epsilon=5e-5,
                                     mode='min')
      
      callbacks_list = [checkpoint, early, lr_reduced]
      
      
      if numIter>=numLimit+1:
            pass
      else:       
            full_df = pd.DataFrame(full_df_COPY)
            tmp = full_df[:len_train]
            train_df = tmp.iloc[train_index]
            del tmp;gc.collect()
                        
            # 不考虑使用均值
            try:
                  full_df.drop('median_deal_probability_price_p', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_price_p', axis=1, inplace=True); gc.collect()
                  full_df.drop('median_deal_probability_param_2', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_param_2', axis=1, inplace=True); gc.collect()
                  full_df.drop('median_deal_probability_item_seq_number_p', axis=1, inplace=True); gc.collect()
                  train_df.drop('median_deal_probability_item_seq_number_p', axis=1, inplace=True); gc.collect()
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
            
            
            from sklearn.cross_validation import KFold
            NFOLDS = 5#5
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
            del ridge_oof_train, ridge_oof_test, ridge_preds, sgd_oof_test, sgd_oof_train, sgd_preds, ready_df
            gc.collect()
                                                
            full_df.drop("item_id", axis=1, inplace=True)
            
            # Combine Dense Features with Sparse Text Bag of Words Features
            X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
            X_test_full=full_df.iloc[len_train:]
            X_test_ready=ready_full_df[len_train:]
#            del ready_full_df, full_df
            gc.collect()        
            
            # mean rmse is: 0.2260609935447737
            print('ridge 3 (full) oof ...')
            ridge_oof_train, ridge_oof_test = get_oof(ridge, X.tocsr(), y, hstack([csr_matrix(full_df.iloc[len_train:]), ready_full_df[len_train:]]).tocsr())
            ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
            tmp = pd.DataFrame(full_df)
            tmp['ridge_preds_3'] = ridge_preds.astype(np.float32)
            tmp['ridge_preds_3'].clip(0.0, 1.0, inplace=True)            
                                            
            full_df['ridge_preds_3'] = tmp['ridge_preds_3'].astype(np.float32)
                     
            del tmp; gc.collect()
   
            # =============================================================================
            # NN feature handle         
            # =============================================================================
            full_df.fillna(-1, inplace=True)
            # categorical cols
            all_cols = list(full_df)
            cat_col = [
                       'region',
                       'city',
                       'parent_category_name',
                       'category_name',
                       'user_type',
                       'image_top_1',
                       'param_1',
                       'param_2',
                       'param_3',
                       'price_p',
                       'item_seq_number_p',
                       ]
            # scale_cols (numerical)
            for col in cat_col:
                  all_cols.remove(col)
            scale_cols  = all_cols
            
            # standarize
            full_df[scale_cols] = preprocessing.scale(full_df[scale_cols])
                        
#            hot_enc_cols_cat = cat_col
#            full_df = pd.get_dummies(full_df, columns=hot_enc_cols_cat)
            
            print("getting keras data ...")
            
            
            def get_keras_sparse(df, ready_full_df):
                  X = {'sparse_data': ready_full_df,
                  'region': np.array(df['region']),
                  'city': np.array(df['city']),
                  'parent_category_name': np.array(df['parent_category_name']),
                  'category_name': np.array(df['category_name']),
                  'user_type': np.array(df['user_type']),
                  'image_top_1': np.array(df['image_top_1']),
                  'param_1': np.array(df['param_1']),
                  'param_2': np.array(df['param_2']),
                  'param_3': np.array(df['param_3']),
                  'price_p': np.array(df['price_p']),
                  'item_seq_number_p': np.array(df['item_seq_number_p']),
                  
                  'median_deal_probability_price_p': np.array(df['median_deal_probability_price_p']),
                  'median_deal_probability_param_2': np.array(df['median_deal_probability_param_2']),
                  'median_deal_probability_item_seq_number_p': np.array(df['median_deal_probability_item_seq_number_p']),
                  
                  'sgd_preds_1': np.array(df['sgd_preds_1']),
                  'sgd_preds_2': np.array(df['sgd_preds_2']),
                  'ridge_preds_1': np.array(df['ridge_preds_1']),
                  'ridge_preds_2': np.array(df['ridge_preds_2']),
                  'ridge_preds_1a': np.array(df['ridge_preds_1a']),
                  'ridge_preds_2a': np.array(df['ridge_preds_2a']),
                  'ridge_preds_3': np.array(df['ridge_preds_3']),
                  
                  'average_HLS_Hs': np.array(df['average_HLS_Hs']),
                  'average_HLS_Ls': np.array(df['average_HLS_Ls']),
                  'average_HLS_Ss': np.array(df['average_HLS_Ss']),
                  'average_HSV_Ss': np.array(df['average_HSV_Ss']),
                  'average_HSV_Vs': np.array(df['average_HSV_Vs']),
                  'average_LUV_Ls': np.array(df['average_LUV_Ls']),
                  'average_LUV_Us': np.array(df['average_LUV_Us']),
                  'average_LUV_Vs': np.array(df['average_LUV_Vs']),
                  'average_YUV_Us': np.array(df['average_YUV_Us']),
                  'average_YUV_Vs': np.array(df['average_YUV_Vs']),
                  'average_YUV_Ys': np.array(df['average_YUV_Ys']),
                  'average_reds': np.array(df['average_reds']),
                  'average_blues': np.array(df['average_blues']),
                  'average_greens': np.array(df['average_greens']),
                  'average_pixel_width': np.array(df['average_pixel_width']),
                  'blurinesses': np.array(df['blurinesses']),
                  'dullnesses': np.array(df['dullnesses']),
                  'heights': np.array(df['heights']),
                  'image_quality': np.array(df['image_quality']),
         
                  
                  'text_feature_num_chars': np.array(df['text_feature_num_chars']),
                  'text_feature_num_words': np.array(df['text_feature_num_words']),
                  'text_feature_num_unique_words': np.array(df['text_feature_num_unique_words']),
                  'text_feature_words_vs_unique': np.array(df['text_feature_words_vs_unique']),
                  'text_feature_2_num_chars': np.array(df['text_feature_2_num_chars']),
                  'text_feature_2_num_words': np.array(df['text_feature_2_num_words']),
                  'text_feature_2_num_unique_words': np.array(df['text_feature_2_num_unique_words']),
                  'text_feature_2_words_vs_unique': np.array(df['text_feature_2_words_vs_unique']),
                  'description_num_chars': np.array(df['description_num_chars']),
                  'description_num_words': np.array(df['description_num_words']),
                  'description_num_unique_words': np.array(df['description_num_unique_words']),
                  'description_words_vs_unique': np.array(df['description_words_vs_unique']),
                  'title_num_chars': np.array(df['title_num_chars']),
                  'title_num_words': np.array(df['title_num_words']),                  
                  'title_num_unique_words': np.array(df['title_num_unique_words']),
                  'title_words_vs_unique': np.array(df['title_words_vs_unique']),
                  
                  
                  'avg_days_up_user': np.array(df['avg_days_up_user']),
                  'avg_times_up_user': np.array(df['avg_times_up_user']),
                  'income': np.array(df['income']),
                  'n_user_items': np.array(df['n_user_items']),
                  'population': np.array(df['population']),
                  'price': np.array(df['price']),
                  'wday': np.array(df['wday']),
  
                                  
                  'num_desc_punct': np.array(df['num_desc_punct']),
                  'num_desc_capE': np.array(df['num_desc_capE']),
                  'num_desc_capP': np.array(df['num_desc_capP']),
                  'num_title_punct': np.array(df['num_title_punct']),
                  'num_title_capE': np.array(df['num_title_capE']),
                  'num_title_capP': np.array(df['num_title_capP']),
                  'is_in_desc_1': np.array(df['is_in_desc_хорошо']),
                  'is_in_desc_2': np.array(df['is_in_desc_Плохо']),
                  'is_in_desc_3': np.array(df['is_in_desc_новый']),
                  'is_in_desc_4': np.array(df['is_in_desc_старый']),
                  'is_in_desc_5': np.array(df['is_in_desc_используемый']),
                  'is_in_desc_6': np.array(df['is_in_desc_есплатная_доставка']),
                  'is_in_desc_7': np.array(df['is_in_desc_есплатный_возврат']),
                  'is_in_desc_8': np.array(df['is_in_desc_идеально']),
                  'is_in_desc_9': np.array(df['is_in_desc_подержанный']),
                  'is_in_desc_10': np.array(df['is_in_desc_пСниженные_цены']),
                  'num_title_Exclamation': np.array(df['num_title_Exclamation']),
                  'num_title_Question': np.array(df['num_title_Question']),
                  'num_desc_Exclamation': np.array(df['num_desc_Exclamation']),
                  'num_desc_Question': np.array(df['num_desc_Question']),                  
                  }
                  
                  return X


            print("sepatating train/ val...")
            
            X_train, X_valid = full_df.iloc[train_index], full_df.iloc[valid_index]
            X_train_ready, X_valid_ready = ready_full_df.tocsr()[train_index],  ready_full_df.tocsr()[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]   
            
            X_test = full_df[len_train:]
            X_test_ready = ready_full_df.tocsr()[len_train:]
            
            X_train = get_keras_sparse(X_train, X_train_ready)
            X_valid = get_keras_sparse(X_valid, X_valid_ready)
            X_test = get_keras_sparse(X_test, X_test_ready)

            def get_callbacks(filepath, patience=4):
                es = EarlyStopping('val_loss', patience=patience, mode="min")# Stop training when a monitored quantity has stopped improving.
                msave = ModelCheckpoint(filepath, save_best_only=True)#saves the best latest model and doesnt let it be overwritten
                return [es, msave]
          

            model = get_model(X_train)
            # model.summary()
                        
            BATCH_SIZE = 128#128
            epochs = 30
            
            hist = model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH_SIZE, 
                             validation_data=(X_valid, y_valid), verbose=2,
                             shuffle=False, callbacks=callbacks_list)#shuffle=False,
            
            model.load_weights(file_path)
            pred_test = model.predict(X_test, batch_size=4000)
            pred_test = pred_test.flatten()
            pred_valid = model.predict(X_valid, batch_size=4000)
            pred_train = model.predict(X_train, batch_size=4000)
            
            print('validation rmse: ', rmse(y_valid, pred_valid.flatten()))
            print('train rmse: ', rmse(y_train, pred_train.flatten()))
            
            val_predict[valid_index] = pred_valid.flatten()
            
            nnsub = pd.DataFrame(pred_test,columns=["deal_probability"],index=sub_item_id)
            nnsub["deal_probability"].clip(0.0, 1.0, inplace=True)
            nnsub.to_csv("ml_nn_sub_{}.csv".format(numIter),index=True, header=True)
            
            
            mean_train_rmse += rmse(y_train, pred_train.flatten())
            mean_valid_rmse += rmse(y_valid, pred_valid.flatten())

print("mean train rmse: ", mean_train_rmse/numLimit)
print("mean valid rmse: ", mean_valid_rmse/numLimit)

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_item_ids.reshape(len(train_user_ids), 1)

val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv('ml_nn_oof.csv', index=False)
'''
1w 5 folds
#mean train rmse:  0.23350688575361983
#mean valid rmse:  0.24647966724067033
'''