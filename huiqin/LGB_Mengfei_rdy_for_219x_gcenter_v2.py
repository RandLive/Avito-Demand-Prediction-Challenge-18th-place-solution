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
import pickle
from sklearn.utils import shuffle
from contextlib import contextmanager
from sklearn.externals import joblib
from math import sin, cos, sqrt, atan2, radians

debug = False
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("input/train.csv",  parse_dates = ["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("input/test.csv",  parse_dates = ["activation_date"])
    # suppl
    train_active = pd.read_csv("input/train_active.csv", usecols=used_cols)
    test_active = pd.read_csv("input/test_active.csv", usecols=used_cols)
    train_periods = pd.read_csv("input/periods_train.csv", parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("input/periods_test.csv", parse_dates=["date_from", "date_to"])
else:
    train_df = pd.read_csv("input/train.csv", parse_dates = ["activation_date"])
    train_df = shuffle(train_df, random_state=1234); train_df = train_df.iloc[:5000]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("input/test.csv",  nrows=1000, parse_dates = ["activation_date"])
    # suppl 
    train_active = pd.read_csv("input/train_active.csv",  nrows=1000, usecols=used_cols)
    test_active = pd.read_csv("input/test_active.csv",  nrows=1000, usecols=used_cols)
    train_periods = pd.read_csv("input/periods_train.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("input/periods_test.csv",  nrows=1000, parse_dates=["date_from", "date_to"])
print("loading data done!")

# =============================================================================
# add geo info: https://www.kaggle.com/frankherfert/avito-russian-region-cities/data
# =============================================================================

"""
tmp = pd.read_csv("input/avito_region_city_features.csv", usecols=["city", "region", "city_region", "latitude", "longitude"])

#TODO city distance to geo center 
def get_distance(lati, longi, to):
    R = 6373.0
    if to == "Краснодар":
        geo_center = [45.0393, 38.9872] #The most popular city on avito
    if to == "Екатеринбург":
        geo_center = [56.8389, 60.6057] #The 2nd popular city on avito    
    if to == "Новосибирск":
        geo_center = [55.0084, 82.9357] #The 3rd popular city on avito    
    if to == "Ростов-на-Дону":
        geo_center = [47.2357, 39.7015] #The 4th popular city on avito    
    if to == "Нижний Новгород":
        geo_center = [56.2965, 43.9361] #The 5th popular city on avito
    if to == "Челябинск":
        geo_center = [55.1644, 61.4368] #The 5th popular city on avito
    if to == "Пермь":
        geo_center = [58.0297, 56.2668] #The 5th popular city on avito
    if to == "Казань":
        geo_center = [55.8304, 49.0661] #The 5th popular city on avito
    if to == "Самара":
        geo_center = [53.2415, 50.2212] #The 5th popular city on avito
    if to == "Омск":
        geo_center = [54.9885, 73.3242] #The 5th popular city on avito
    
    lat1 = radians(lati)
    lon1 = radians(longi)
    lat2 = radians(geo_center[0])
    lon2 = radians(geo_center[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance
print("getting distance to geo center..")

tmp["to_center_dis_1"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_1"])):
    tmp["to_center_dis_1"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Краснодар")

tmp["to_center_dis_2"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_2"])):
    tmp["to_center_dis_2"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Екатеринбург")
    
tmp["to_center_dis_3"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_3"])):
    tmp["to_center_dis_3"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Новосибирск")
    
tmp["to_center_dis_4"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_4"])):
    tmp["to_center_dis_4"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Ростов-на-Дону")
    
tmp["to_center_dis_5"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_5"])):
    tmp["to_center_dis_5"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Нижний Новгород")
    
tmp["to_center_dis_6"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_6"])):
    tmp["to_center_dis_6"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Челябинск")
    
tmp["to_center_dis_7"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_7"])):
    tmp["to_center_dis_7"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Пермь")
    
tmp["to_center_dis_8"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_8"])):
    tmp["to_center_dis_8"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Казань")
    
tmp["to_center_dis_9"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_9"])):
    tmp["to_center_dis_9"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Самара")
    
tmp["to_center_dis_10"] = tmp["city"].copy()
for index in range(len(tmp["to_center_dis_10"])):
    tmp["to_center_dis_10"][index] = get_distance(tmp["latitude"][index], tmp["longitude"][index], to = "Омск")
    
print('Saving pickled data')
with open('avito_region_city_features_gcenter_v2.p', 'wb') as f:
    pickle.dump(tmp, f)
"""
print('Loading pickled data')
with open('avito_region_city_features_gcenter_v2.p', 'rb') as f:
    tmp = pickle.load(f)

train_df = train_df.merge(tmp, on=["city","region"], how="left")
train_df["lat_long"] = train_df["latitude"]+train_df["longitude"]
test_df = test_df.merge(tmp, on=["city","region"], how="left")
test_df["lat_long"] = test_df["latitude"]+test_df["longitude"]

del tmp; gc.collect()

# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("input/region_income.csv", sep=";", names=["region", "income"])
#tmp["income"] = np.log1p(tmp["income"])
train_df = train_df.merge(tmp, on="region", how="left")
test_df = test_df.merge(tmp, on="region", how="left")
del tmp; gc.collect()
# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("input/city_population_wiki_v3.csv")
#tmp["population"] = np.log1p(tmp["population"])
train_df = train_df.merge(tmp, on="city", how="left")
test_df = test_df.merge(tmp, on="city", how="left")
del tmp; gc.collect()
# =============================================================================
# Add image quality: by steeve
# ============================================================================= 
with open('input/inception_v3_include_head_max_train.p','rb') as f:
    x = pickle.load(f)
    
train_features = x['features']
train_ids = x['ids']

with open('input/inception_v3_include_head_max_test.p','rb') as f:
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

   
with open('input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_blurinesses = x['blurinesses']
train_ids = x['ids']

with open('input/test_image_features.p','rb') as f:
    x = pickle.load(f)

test_whitenesses = x['blurinesses']
test_ids = x['ids']    
del x; gc.collect()

incep_train_image_df = pd.DataFrame(train_blurinesses, columns = ['blurinesses'])
incep_test_image_df = pd.DataFrame(test_whitenesses, columns = [f'blurinesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


print('adding whitenesses ...')
with open('input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_whitenesses = x['whitenesses']
train_ids = x['ids']

with open('input/test_image_features.p','rb') as f:
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
with open('input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_dullnesses = x['dullnesses']
train_ids = x['ids']

with open('input/test_image_features.p','rb') as f:
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
del gp_df; gc.collect()

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
        
        # 0527
        df["dayofyear"] = df["activation_date"].dt.dayofyear
        
    def Do_Label_Enc(df):
        print("feature engineering -> lable encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
               "category_name", "user_type", "image_top_1",
               "param_1", "param_2", "param_3","image",
               "city_region",
               ]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()
        
        # 0526
        df['param_12'] = df['param_1']*df['param_2']
        df['param_13'] = df['param_1']*df['param_3']
        df['param_23'] = df['param_2']*df['param_3']
    
    import string
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])         
    def Do_NA(df):
        print("feature engineering -> fill na ...")
#        df["price"] = np.log(df["price"]+0.001).astype("float32")
#        df["price"].fillna(-1,inplace=True)
        df["price_vs_income"] = df["price"] / df["income"]
        df["price_vs_income"].fillna(-1, inplace=True)
        df["image_top_1"].fillna(-1,inplace=True)
        # 0528
#        df["blurinesses"] = np.log(df["blurinesses"]+0.001).astype("float32")
        df["blurinesses"].fillna(-1234,inplace=True)
#        df["whitenesses"] = np.log(df["whitenesses"]+0.001).astype("float32")
        df["whitenesses"].fillna(-1234,inplace=True)
        df["dullnesses"].fillna(-1234,inplace=True)
#        df["image_quality"] = np.log(df["image_quality"]+0.001).astype("float32")        
        df["image_quality"].fillna(-1,inplace=True)
#        df["image_top_4"].fillna(-1,inplace=True)
        df["image"].fillna("noinformation",inplace=True)
        df["param_1"].fillna("nicapotato",inplace=True)
        df["param_2"].fillna("nicapotato",inplace=True)
        df["param_3"].fillna("nicapotato",inplace=True)
        df["title"].fillna("nicapotato",inplace=True)
        df["description"].fillna("nicapotato",inplace=True)
        df["to_center_dis_1"].fillna(-1,inplace=True)
        df["to_center_dis_2"].fillna(-1,inplace=True)
        df["to_center_dis_3"].fillna(-1,inplace=True)
        df["to_center_dis_4"].fillna(-1,inplace=True)
        df["to_center_dis_5"].fillna(-1,inplace=True)
        df["to_center_dis_6"].fillna(-1,inplace=True)
        df["to_center_dis_7"].fillna(-1,inplace=True)
        df["to_center_dis_8"].fillna(-1,inplace=True)
        df["to_center_dis_9"].fillna(-1,inplace=True)
        df["to_center_dis_10"].fillna(-1,inplace=True)
        
    def Do_Count(df):  
        print("feature engineering -> do count ...")
        # some count       
        df["num_desc_punct"] = df["description"].apply(lambda x: count(x, set(string.punctuation)))
        df["num_desc_capE"] = df["description"].apply(lambda x: count(x, "[A-Z]"))
        df["num_desc_capP"] = df["description"].apply(lambda x: count(x, "[А-Я]"))
        
        df["num_title_punct"] = df["title"].apply(lambda x: count(x, set(string.punctuation)))
        df["num_title_capE"] = df["title"].apply(lambda x: count(x, "[A-Z]"))
        df["num_title_capP"] = df["title"].apply(lambda x: count(x, "[А-Я]"))               
        # good, used, bad ... count
        df["is_in_desc_хорошо"] = df["description"].str.contains("хорошо").map({True:1, False:0})
        df["is_in_desc_Плохо"] = df["description"].str.contains("Плохо").map({True:1, False:0})
        df["is_in_desc_новый"] = df["description"].str.contains("новый").map({True:1, False:0})
        df["is_in_desc_старый"] = df["description"].str.contains("старый").map({True:1, False:0})
        df["is_in_desc_используемый"] = df["description"].str.contains("используемый").map({True:1, False:0})
        df["is_in_desc_есплатная_доставка"] = df["description"].str.contains("есплатная доставка").map({True:1, False:0})
        df["is_in_desc_есплатный_возврат"] = df["description"].str.contains("есплатный возврат").map({True:1, False:0})
        df["is_in_desc_идеально"] = df["description"].str.contains("идеально").map({True:1, False:0})
        df["is_in_desc_подержанный"] = df["description"].str.contains("подержанный").map({True:1, False:0})
        df["is_in_desc_пСниженные_цены"] = df["description"].str.contains("пСниженные_цены").map({True:1, False:0})
 #       df["is_in_desc_идеальный"] = df["description"].str.contains("идеальный").map({True:1, False:0})
 #       df["is_in_desc_красивая"] = df["description"].str.contains("красивая").map({True:1, False:0})
 #       df["is_in_desc_хороший"] = df["description"].str.contains("хороший").map({True:1, False:0})
 #       df["is_in_desc_гарантированный"] = df["description"].str.contains("гарантированный").map({True:1, False:0})
 #       df["is_in_desc_замечательно"] = df["description"].str.contains("замечательно").map({True:1, False:0})
 #       df["is_in_desc_сделка"] = df["description"].str.contains("сделка").map({True:1, False:0})
 #       df["is_in_desc_отличный"] = df["description"].str.contains("отличный").map({True:1, False:0})
 #       df["is_in_desc_великолепный"] = df["description"].str.contains("великолепный").map({True:1, False:0})
 #       df["is_in_desc_выдающийся"] = df["description"].str.contains("выдающийся").map({True:1, False:0})
 #       df["is_in_desc_полезный"] = df["description"].str.contains("полезный").map({True:1, False:0})
 #       df["is_in_desc_почти_новый"] = df["description"].str.contains("почти_новый").map({True:1, False:0})
 #       df["is_in_desc_выгодная_покупка"] = df["description"].str.contains("выгодная покупка").map({True:1, False:0})
 #       df["is_in_desc_удачная_покупка"] = df["description"].str.contains("удачная покупка").map({True:1, False:0})
 #       df["is_in_desc_дешево_купленная_вещь"] = df["description"].str.contains("дешево купленная вещь").map({True:1, False:0})
                           
    def Do_Drop(df):
        df.drop(["activation_date"], axis=1, inplace=True)
        
    def Do_Stat_Text(df):
        print("feature engineering -> statistics in text ...")
        # 0527
        textfeats = ["text_feature_2","description", "title"]
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
    "smooth_idf":False}
    
#    tfidf_para_2 = {
#    "stop_words": russian_stop,
#    "analyzer": "char",
#    "token_pattern": r"\w{1,}",
#    "sublinear_tf": True,
#    "dtype": np.float32,
#    "norm": "l2",
#    #"min_df":5,
#    #"max_df":.9,
#    "smooth_idf":False}
    
    def get_col(col_name): return lambda x: x[col_name]
    vectorizer = FeatureUnion([
        ("description", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col("description"))
         ),
        ("title_description", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=18000,
            **tfidf_para,
            preprocessor=get_col("title_description"))
         ),
        ("text_feature", CountVectorizer(
            ngram_range=(1, 2),
            preprocessor=get_col("text_feature"))
         ),      
        ("title", TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            preprocessor=get_col("title"))
         ),    
#        ("title_2", TfidfVectorizer(
#            ngram_range=(1, 6),
#            **tfidf_para_2,
#            preprocessor=get_col("title"))
#         ),
    ])
      
    vectorizer.fit(df.to_dict("records"))
    ready_full_df = vectorizer.transform(df.to_dict("records"))    
    tfvocab = vectorizer.get_feature_names()    
    df.drop(["text_feature", "text_feature_2", "description", "title", "title_description"], axis=1, inplace=True)
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
    df["price+"] = np.round(df["price"]*4.8).astype(int)   
    df["item_seq_number+"] = np.round(df["item_seq_number"]/100).astype(int)     
    return df

# label encoding
train_df, val_df = train_test_split(
    full_df.iloc[:len_train], test_size=0.1, random_state=42) #23    
del val_df; gc.collect()
def feature_Eng_On_Deal_Prob(df, df_train):
    print('feature engineering -> on price deal prob +...')
    df2 = df    
    tmp = df_train.groupby(["price+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_price+'})     
    df = pd.merge(df, tmp, how='left', on=["price+"])
    df2['median_deal_probability_price+'] = df['median_deal_probability_price+']  
    del tmp; gc.collect()
    
    tmp = df_train.groupby(["item_seq_number+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_item_seq_number+'})     
    df = pd.merge(df, tmp, how='left', on=["item_seq_number+"])
    df2['median_deal_probability_item_seq_number+'] = df['median_deal_probability_item_seq_number+']
    
    # 0526
    tmp = df_train.groupby(["param_2"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_param_2'})     
    df = pd.merge(df, tmp, how='left', on=["param_2"])
    df2['median_deal_probability_param_2'] = df['median_deal_probability_param_2']
    
    # 0529
    tmp = df_train.groupby(["wday"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_wday'})     
    df = pd.merge(df, tmp, how='left', on=["wday"])
    df2['median_deal_probability_wday'] = df['median_deal_probability_wday']
    
    # 0527
    tmp = df.groupby(["city"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_city'})     
    df = pd.merge(df, tmp, how='left', on=["city"])
    df2['count_user_id_city'] = df['count_user_id_city']
    df2['count_user_id_city_vs_population'] = df2['count_user_id_city'] / df2['population'] * 1000000
    del df2['count_user_id_city']
#      
    # 0528 : 0.233263
    tmp = df.groupby(["region"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_region'})     
    df = pd.merge(df, tmp, how='left', on=["region"])
    df2['count_user_id_region'] = df['count_user_id_region']
    
    tmp = df.groupby(["wday"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_wday'})     
    df = pd.merge(df, tmp, how='left', on=["wday"])
    df2['count_user_id_wday'] = df['count_user_id_wday']

    tmp = df.groupby(["price+"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_price+'})     
    df = pd.merge(df, tmp, how='left', on=["price+"])
    df2['count_user_id_price+'] = df['count_user_id_price+']
    
    tmp = df.groupby(["param_2"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_param_2'})     
    df = pd.merge(df, tmp, how='left', on=["param_2"])
    df2['count_user_id_param_2'] = df['count_user_id_param_2']
        
    tmp = df.groupby(["image_top_1"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_image_top_1'})     
    df = pd.merge(df, tmp, how='left', on=["image_top_1"])
    df2['count_user_id_image_top_1'] = df['count_user_id_image_top_1']
   
    tmp = df.groupby(["user_type"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_user_type'})     
    df = pd.merge(df, tmp, how='left', on=["user_type"])
    df2['count_user_id_user_type'] = df['count_user_id_user_type']

    tmp = df.groupby(["parent_category_name"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_parent_category_name'})     
    df = pd.merge(df, tmp, how='left', on=["parent_category_name"])
    df2['count_user_id_parent_category_name'] = df['count_user_id_parent_category_name']
         
    tmp = df.groupby(["item_seq_number+"], as_index=False)['user_id'].count().rename(columns={'user_id':'count_user_id_item_seq_number+'})     
    df = pd.merge(df, tmp, how='left', on=["item_seq_number+"])
    df2['count_user_id_item_seq_number+'] = df['count_user_id_item_seq_number+']
          
    df2.drop("item_id", axis=1, inplace=True)

    df2.fillna(-1, inplace=True)    
    del tmp; gc.collect()
    return df2

del full_df['deal_probability']; gc.collect()

# =============================================================================
# use additianl image data
# =============================================================================
feature_engineering(full_df)
feature_engineering(train_df)

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

#print("hstack 1 ...")
## Combine Dense Features with Sparse Text Bag of Words Features
#X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
#test = hstack([csr_matrix(full_df.iloc[len_train:]), ready_full_df[len_train:]]) # Sparse Matrix
#
#print('ridge 3 oof ...')
#ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
#                'max_iter':10, 'tol':0.001, 'solver':'sag', 'random_state':SEED}
#ridge_oof_train, ridge_oof_test = get_oof(ridge, X.tocsr()[:len_train], y, test)
#ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
#full_df['ridge_preds_3'] = ridge_preds
#full_df['ridge_preds_3'].clip(0.0, 1.0, inplace=True)
#del X, test; gc.collect()


# optimize ram usage
list_float = ["avg_days_up_user", "avg_times_up_user", "price", "population",
             "price_vs_income","text_feature_2_words_vs_unique","blurinesses","whitenesses","dullnesses","image_quality",
             "description_words_vs_unique", "title_words_vs_unique", 
             "median_deal_probability_price+", "median_deal_probability_item_seq_number+", "median_deal_probability_param_2", 
             "median_deal_probability_wday",
             "count_user_id_city_vs_population", "ridge_preds_1", "ridge_preds_2",
             "count_user_id_region", "count_user_id_wday", "count_user_id_price+",
             "count_user_id_param_2", "count_user_id_image_top_1", "count_user_id_user_type",
             "count_user_id_parent_category_name", "count_user_id_item_seq_number+",            
             ]
list_int = [list_tmp for list_tmp in list(full_df) if list_tmp not in list_float]
full_df[list_float] = full_df[list_float].astype('float32')
full_df[list_int] = full_df[list_int].astype('int32')
gc.collect()


del ridge_preds, ridge_oof_train, ridge_oof_test; gc.collect()
print("hstack 2 ...")
print("Modeling Stage ...")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]]) # Sparse Matrix
test = hstack([csr_matrix(full_df.iloc[len_train:]), ready_full_df[len_train:]]) # Sparse Matrix

for shape in [X,test]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))

joblib.dump(test, "test.pkl")
del test; gc.collect()

tfvocab = full_df.columns.tolist() + tfvocab
del full_df, ready_full_df; gc.collect()

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

print("save X,y ...")
joblib.dump(X, "X.pkl")
joblib.dump(y, "y.pkl")
joblib.dump(tfvocab, "tfvocab.pkl")

"""
print("load X,y ...")
X = joblib.load("X.pkl")
y = joblib.load("y.pkl")
tfvocab = joblib.load("tfvocab.pkl")
"""

for numIter in range(0, 1):
      
      X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42) #23
      del X, y;
      gc.collect()
      
#      X_train, X_valid = X.tocsr()[train_index], X.tocsr()[test_index]
#      y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
     
      lgbm_params =  {
              "tree_method": "feature",    
              "num_threads": 6,
              "task": "train",
              "boosting_type": "gbdt",
              "objective": "regression",
              "metric": "rmse",
 #             "max_depth": 15, 
              "num_leaves": 420, # 360
              "feature_fraction": 0.2, #0.2
              "bagging_fraction": 0.2, #0.2
              "learning_rate": 0.015,
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
      
      del X_train, y_train; gc.collect()
      
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
      
      
      test = joblib.load("test.pkl")
      
      print("Model Evaluation Stage")
      print( "RMSE:", rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)) )
      lgpred = lgb_clf.predict(test, num_iteration=lgb_clf.best_iteration)
      
      lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=sub_item_id)
      lgsub["deal_probability"].clip(0.0, 1.0, inplace=True) # Between 0 and 1
      lgsub.to_csv("ml_lgb_sub_{}.csv".format(numIter),index=True,header=True)

      rmse_sume += rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration))
      
      numIter += 1
      
      del X_valid, y_valid, lgtrain, lgvalid, lgpred, lgsub, test, sub_item_id
      gc.collect()

print("mean rmse is:", rmse_sume/numIter)
      
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
[100]   train's rmse: 0.224799  valid's rmse: 0.226921
[200]   train's rmse: 0.216856  valid's rmse: 0.221171
[300]   train's rmse: 0.212721  valid's rmse: 0.219112
[400]   train's rmse: 0.209271  valid's rmse: 0.217955


[3662]  train's rmse: 0.171134  valid's rmse: 0.214972

"""