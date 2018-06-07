# encoding=utf-8
'''
20w数据的快速测试验证特征版本
'''
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
import time
start_time=time.time()
print("Starting job at time:", time.time())
debug = True
print("loading data ...")
used_cols = ["item_id", "user_id"]
if debug == False:
    train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv", parse_dates=["activation_date"])

    # 加入翻译的title文本
    train_df['title_ru'] = pd.read_csv('../input/train_ru_title.csv')
    test_df['title_ru'] = pd.read_csv('../input/test_ru_title.csv')
    # suppl
    train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=["date_from", "date_to"])
else:
    train_df = pd.read_csv("../input/train.csv", parse_dates=["activation_date"])
    train_df = shuffle(train_df, random_state=1234);
    # 加入翻译的title文本
    train_df['title_ru'] = pd.read_csv('../input/train_ru_title.csv')

    train_df = train_df.iloc[:200000]
    y = train_df["deal_probability"]
    test_df = pd.read_csv("../input/test.csv", nrows=1000, parse_dates=["activation_date"])
    test_df['title_ru'] = pd.read_csv('../input/test_ru_title.csv', nrows=1000)
    # suppl
    train_active = pd.read_csv("../input/train_active.csv", nrows=1000, usecols=used_cols)
    test_active = pd.read_csv("../input/test_active.csv", nrows=1000, usecols=used_cols)
    train_periods = pd.read_csv("../input/periods_train.csv", nrows=1000, parse_dates=["date_from", "date_to"])
    test_periods = pd.read_csv("../input/periods_test.csv", nrows=1000, parse_dates=["date_from", "date_to"])
print("loading data done!")

# =============================================================================
# Add image quality: by steeve
# =============================================================================
import pickle

with open('../input/inception_v3_include_head_max_train.p', 'rb') as f:
    x = pickle.load(f)

train_features = x['features']
train_ids = x['ids']

with open('../input/inception_v3_include_head_max_test.p', 'rb') as f:
    x = pickle.load(f)

test_features = x['features']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_features, columns=['image_quality'])
incep_test_image_df = pd.DataFrame(test_features, columns=[f'image_quality'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)

train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')


with open('../input/train_image_features.p', 'rb') as f:
    x = pickle.load(f)

train_blurinesses = x['blurinesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
    x = pickle.load(f)

test_blurinesses = x['blurinesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_blurinesses, columns=['blurinesses'])
incep_test_image_df = pd.DataFrame(test_blurinesses, columns=[f'blurinesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')

print('adding whitenesses ...')
with open('../input/train_image_features.p', 'rb') as f:
    x = pickle.load(f)

train_whitenesses = x['whitenesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
    x = pickle.load(f)

test_whitenesses = x['whitenesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_whitenesses, columns=['whitenesses'])
incep_test_image_df = pd.DataFrame(test_whitenesses, columns=[f'whitenesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')

print('adding dullnesses ...')
with open('../input/train_image_features.p', 'rb') as f:
    x = pickle.load(f)

train_dullnesses = x['dullnesses']
train_ids = x['ids']

with open('../input/test_image_features.p', 'rb') as f:
    x = pickle.load(f)

test_dullnesses = x['dullnesses']
test_ids = x['ids']
del x;
gc.collect()

incep_train_image_df = pd.DataFrame(train_dullnesses, columns=['dullnesses'])
incep_test_image_df = pd.DataFrame(test_dullnesses, columns=[f'dullnesses'])
incep_train_image_df['image'] = (train_ids)
incep_test_image_df['image'] = (test_ids)
train_df = train_df.join(incep_train_image_df.set_index('image'), on='image')
test_df = test_df.join(incep_test_image_df.set_index('image'), on='image')

# =============================================================================
# add geo info: https://www.kaggle.com/frankherfert/avito-russian-region-cities/data
# =============================================================================
# tmp = pd.read_csv("../input/avito_region_city_features.csv", usecols=["region", "city", "latitude","longitude"])
# train_df = train_df.merge(tmp, on=["city","region"], how="left")
# train_df["lat_long"] = train_df["latitude"]+train_df["longitude"]
# test_df = test_df.merge(tmp, on=["city","region"], how="left")
# test_df["lat_long"] = test_df["latitude"]+test_df["longitude"]
# del tmp; gc.collect()

# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/region_income.csv", sep=";", names=["region", "income"])
train_df = train_df.merge(tmp, on="region", how="left")
test_df = test_df.merge(tmp, on="region", how="left")
del tmp;
gc.collect()
# =============================================================================
# Add region-income
# =============================================================================
tmp = pd.read_csv("../input/city_population_wiki_v3.csv")
train_df = train_df.merge(tmp, on="city", how="left")
test_df = test_df.merge(tmp, on="city", how="left")
del tmp;
gc.collect()

# =============================================================================
# Here Based on https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm/code
# =============================================================================
all_samples = pd.concat([train_df, train_active, test_df, test_active]).reset_index(drop=True)
all_samples.drop_duplicates(["item_id"], inplace=True)
del train_active, test_active;
gc.collect()

all_periods = pd.concat([train_periods, test_periods])
del train_periods, test_periods;
gc.collect()

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

gp = all_periods.groupby(["user_id"])[["days_up_sum", "times_put_up"]].mean().reset_index() \
    .rename(index=str, columns={"days_up_sum": "avg_days_up_user",
                                "times_put_up": "avg_times_up_user"})

n_user_items = all_samples.groupby(["user_id"])[["item_id"]].count().reset_index() \
    .rename(index=str, columns={"item_id": "n_user_items"})
gp = gp.merge(n_user_items, on="user_id", how="outer")  # left

del all_samples, all_periods, n_user_items
gc.collect()

train_df = train_df.merge(gp, on="user_id", how="left")
test_df = test_df.merge(gp, on="user_id", how="left")

agg_cols = list(gp.columns)[1:]

del gp;
gc.collect()

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


def read_stopwords():
    with open('RussianStopWords.txt',encoding='utf-8') as fin:
        words=[]
        for line in fin:
            words.append(line.strip())
    return set(words)

def text_preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"(\\u[0-9A-Fa-f]+)", r"", text)
    text = re.sub(r"===", r" ", text)
    # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
    text = " ".join(map(str.strip, re.split('(\d+)', text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = " ".join(text.split())
    return text

def text_preprocessing_v2(text):
    '''
    新的预处理函数
    :param text:
    :return:
    '''
    text = str(text)
    text = text.lower()
    text = re.sub(r'\\xa0', ' ', text)
    text = re.sub(r'\●', ' ', text)
    text = re.sub(r'\😎', ' ', text)
    text = re.sub(r'\👍', ' ', text)
    text = re.sub(r'\»', ' ', text)
    text = re.sub(r'\«', ' ', text)
    text = re.sub(r'\↓', ' ', text)
    text = re.sub(r'iphone', ' iphone ', text)
    text = re.sub(r'samsung', ' samsung ', text)
    text = re.sub(r'apple', ' apple ', text)
    text = re.sub(r'dell', ' dell ', text)
    text = re.sub(r'seilmann', ' steilmann ', text)
    text = re.sub(r'multipad', ' multipad ', text)
    text = re.sub(r'triple', ' triple ', text)
    text = re.sub(r'philip', ' philip ', text)


    #以上0.0001的改进
    # 再次改进0.0001
    text = re.sub(r'ipod', ' ipod ', text)

    # text = re.sub(r'ip4200', ' canon4200 ', text)
    # text = re.sub(r'ip4300', ' canon4300 ', text)
    # text = re.sub(r'ip4500', ' canon4500 ', text)
    # text = re.sub(r'mp500', ' canon500 ', text)
    # text = re.sub(r'mp530', ' canon530 ', text)
    # text = re.sub(r'mp610', ' canon610 ', text)
    # #以上没有影响
    # #
    #
    # text = re.sub(r'hamburg', ' hamburg ', text)
    # text = re.sub(r'lumia', ' lumia ', text)
    # text = re.sub(r'seagate', ' seagate ', text)
    #
    # text = re.sub(r'512mb', ' 512mb ', text)
    # text = re.sub(r'128mb', ' 128mb ', text)
    # text = re.sub(r'256mb', ' 256mb ', text)
    # text = re.sub(r'16gb', ' 16gb ', text)
    # text = re.sub(r'32gb', ' 32gb ', text)
    # text = re.sub(r'64gb', ' 64gb ', text)
    # text = re.sub(r'500gb', ' 500gb ', text)
    # text = re.sub(r'260gb', ' 260gb ', text)
    # text = re.sub(r'250gb', ' 250gb ', text)
    # text = re.sub(r'320gb', ' 320gb ', text)
    # text = re.sub(r'1000gb', ' 1000gb ', text)
    # text = re.sub(r'20gb', ' 20gb ', text)


    #
    # text = re.sub(r'\®', ' ', text)
    # text = re.sub(r'intel', ' intel ', text)
    #
    # text = re.sub(r'canon', ' canon ', text)
    # text = re.sub(r'adidas', ' adidas ', text)
    # text = re.sub(r'gucci', ' gucci ', text)
    # #没有什么改进，不变
    # text = re.sub(r'\\u200b', '  ', text)
    # text = re.sub(r'\\u200d', '  ', text)


    # text = re.sub(r'\квартира', ' \квартира  ', text)
    # text = re.sub(r'nokia', ' nokia ', text)
    # text = re.sub(r'sony', ' sony ', text)
    # text = re.sub(r'xiaomi', ' xiaomi ', text)
    text = re.sub(r'asusintel', ' asus intel ', text)
    text = re.sub(r'00asus', ' asus ', text)
    text = re.sub(r'chevrolet', ' chevrolet ', text)

    text = re.sub(r'nikenike', ' nike ', text)

    #panasoni,0.236955
    text = re.sub(r'\™', ' ', text)
    # text = re.sub(r'panasoni', ' panasonic ', text)
    #mean rmse is: 0.2369177999350502
    text = re.sub(r'compac', ' compac ', text)


    # text = re.sub(r'tomy', ' tomy ', text)
    # text = re.sub(r'✔', ' ', text)
    # text = re.sub(r'👌', ' ', text)
    # text = re.sub(r'💰', ' ', text)
    # text = re.sub(r'❤', ' ', text)
    # text = re.sub(r'htc', ' htc ', text)

    #
    # text = re.sub(r'playstation', ' playstation ', text)
    #
    # text = re.sub(r'huawei', ' huawei ', text)
    #
    # text = re.sub(r'motorola', ' motorola ', text)
    # text = re.sub(r'meizu', ' meizu ', text)
    # text = re.sub(r'nikon', ' nikon ', text)
    #
    # #
    # text = re.sub(r'toshiba', ' toshiba ', text)


    text = re.sub(r'gtx', ' gtx ', text)
    text = re.sub(r"(\\u[0-9A-Fa-f]+)",r"", text)
    text = re.sub(r"===",r" ", text)
    # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = " ".join(text.split())
    return text

def split_rus(text):
    tmp=[]
    # isEn=False
    is_before_en=False
    for i,w in enumerate(text):
        # print(w)
        if ord(w)<256:#如果是英语,
            # isEn=True
            if not is_before_en and i>1:#如果前面字符不是英语，就用空格分隔
                # print('1分隔...')
                tmp.append(' ')
                is_before_en = True
            else:
                is_before_en=True
        else:#如果不是英语
            if is_before_en:
                # print('2分隔...')
                tmp.append(' ')
                is_before_en = False

        tmp.append(w)

    return ''.join(tmp)

@contextmanager
def feature_engineering(df):
    # All the feature engineering here

    def Do_Text_Hash(df):
        print("feature engineering -> hash text ...")
        df["text_feature"] = df.apply(lambda row: " ".join([str(row["param_1"]),
                                                            str(row["param_2"]), str(row["param_3"])]), axis=1)

        df["text_feature_2"] = df.apply(lambda row: " ".join([str(row["param_2"]), str(row["param_3"])]), axis=1)
        # df["title_description"] = df.apply(lambda row: " ".join([str(row["title"]), str(row["description"])]), axis=1)

        print("feature engineering -> preprocess text ...")
        df["text_feature"] = df["text_feature"].apply(lambda x: text_preprocessing(x))
        df["text_feature_2"] = df["text_feature_2"].apply(lambda x: text_preprocessing(x))
        df["description"] = df["description"].apply(lambda x: text_preprocessing(x))
        df["title"] = df["title"].apply(lambda x: text_preprocessing(x))
        # df["title_description"] = df["title_description"].apply(lambda x: text_preprocessing_v2(x))

        # new feature
        # df["description"] = df["description"].apply(lambda x: split_rus(x))
        # df["title"] = df["title"].apply(lambda x: split_rus(x))

    def Do_Datetime(df):
        print("feature engineering -> date time ...")
        df["wday"] = df["activation_date"].dt.weekday
        df["wday"] = df["wday"].astype(np.uint8)

    def Do_Label_Enc(df):
        print("feature engineering -> lable encoding ...")
        lbl = LabelEncoder()
        cat_col = ["user_id", "region", "city", "parent_category_name",
                   "category_name", "user_type", "image_top_1",
                   "param_1", "param_2", "param_3", "image",
                   ]
        for col in cat_col:
            df[col] = lbl.fit_transform(df[col].astype(str))
            gc.collect()

    import string
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])

    def Do_NA(df):
        print("feature engineering -> fill na ...")

        df["image_top_1"].fillna(-1, inplace=True)
        df["image"].fillna("noinformation", inplace=True)
        df["param_1"].fillna("nicapotato", inplace=True)
        df["param_2"].fillna("nicapotato", inplace=True)
        df["param_3"].fillna("nicapotato", inplace=True)
        df["title"].fillna("nicapotato", inplace=True)
        df["description"].fillna("nicapotato", inplace=True)
        # price vs income

    #        df["price_vs_city_income"] = df["price"] / df["income"]
    #        df["price_vs_city_income"].fillna(-1, inplace=True)

    def Do_Count(df):
        print("feature engineering -> do count ...")
        # some count
        df["num_desc_punct"] = df["description"].apply(lambda x: count(x, set(string.punctuation))).astype(np.uint16)
        df["num_desc_capE"] = df["description"].apply(lambda x: count(x, "[A-Z]")).astype(np.uint16)
        df["num_desc_capP"] = df["description"].apply(lambda x: count(x, "[А-Я]")).astype(np.uint16)

        df["num_title_punct"] = df["title"].apply(lambda x: count(x, set(string.punctuation))).astype(np.uint16)
        df["num_title_capE"] = df["title"].apply(lambda x: count(x, "[A-Z]")).astype(np.uint16)
        df["num_title_capP"] = df["title"].apply(lambda x: count(x, "[А-Я]")).astype(np.uint16)
        # good, used, bad ... count
        df["is_in_desc_хорошо"] = df["description"].str.contains("хорошо").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_Плохо"] = df["description"].str.contains("Плохо").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_новый"] = df["description"].str.contains("новый").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_старый"] = df["description"].str.contains("старый").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_используемый"] = df["description"].str.contains("используемый").map({True: 1, False: 0}).astype(
            np.uint8)
        df["is_in_desc_есплатная_доставка"] = df["description"].str.contains("есплатная доставка").map(
            {True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_есплатный_возврат"] = df["description"].str.contains("есплатный возврат").map(
            {True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_идеально"] = df["description"].str.contains("идеально").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_подержанный"] = df["description"].str.contains("подержанный").map({True: 1, False: 0}).astype(
            np.uint8)
        df["is_in_desc_пСниженные_цены"] = df["description"].str.contains("Сниженные цены").map(
            {True: 1, False: 0}).astype(np.uint8)
        #new features
        df["is_in_desc_iphone"] = df["title"].str.contains("iphone").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_ipod"] = df["title"].str.contains("ipod").map({True: 1, False: 0}).astype(np.uint8)

        df["is_in_desc_samsung"] = df["title"].str.contains("samsung").map({True: 1, False: 0}).astype(np.uint8)

        #new again
        # df["is_in_desc_philip"] = df["title"].str.contains("philip").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_canon"] = df["title"].str.contains("canon").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_gtx"] = df["title"].str.contains("gtx").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_gucci"] = df["title"].str.contains("gucci").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_adidas"] = df["title"].str.contains("adidas").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_panasonic"] = df["title"].str.contains("panasonic").map({True: 1, False: 0}).astype(np.uint8)

        #加这块有效
        # df["is_in_desc_intel"] = df["title"].str.contains("intel").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_nokia"] = df["title"].str.contains("nokia").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_sony"] = df["title"].str.contains("sony").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_xiaomi"] = df["title"].str.contains("xiaomi").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_asus"] = df["title"].str.contains("asus").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_playstation"] = df["title"].str.contains("playstation").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_nokia"] = df["title"].str.contains("nokia").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_motorola"] = df["title"].str.contains("motorola").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_meizu"] = df["title"].str.contains("meizu").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_nikon"] = df["title"].str.contains("nikon").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_toshiba"] = df["title"].str.contains("toshiba").map({True: 1, False: 0}).astype(np.uint8)

        df["is_in_desc_lada"] = df["title_ru"].str.contains("lada").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_cht"] = df["title_ru"].str.contains("снт").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_sony"] = df["title_ru"].str.contains("sony").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_asus"] = df["title_ru"].str.contains("asus").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_adidas"] = df["title_ru"].str.contains("adidas").map({True: 1, False: 0}).astype(np.uint8)
        # df["is_in_desc_galaxy"] = df["title_ru"].str.contains("galaxy").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_lg"] = df["title_ru"].str.contains("lg").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_lenovo"] = df["title_ru"].str.contains("lenovo").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_samara"] = df["title_ru"].str.contains("samara").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_toyota"] = df["title_ru"].str.contains("toyota").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_zara"] = df["title_ru"].str.contains("zara").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_hp"] = df["title_ru"].str.contains("hp").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_nokia"] = df["title_ru"].str.contains("nokia").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_chevrolet"] = df["title_ru"].str.contains("chevrolet").map({True: 1, False: 0}).astype(np.uint8)
        #1000以下
        df["is_in_desc_hyundai"] = df["title_ru"].str.contains("hyundai").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_honda"] = df["title_ru"].str.contains("honda").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_nissan"] = df["title_ru"].str.contains("nissan").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_nike"] = df["title_ru"].str.contains("nike").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_volkswagen"] = df["title_ru"].str.contains("volkswagen").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_daewoo"] = df["title_ru"].str.contains("daewoo").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_benz"] = df["title_ru"].str.contains("mercedes-benz").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_xbox"] = df["title_ru"].str.contains("xbox").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_daewoo"] = df["title_ru"].str.contains("daewoo").map({True: 1, False: 0}).astype(np.uint8)
        df["is_in_desc_philips"] = df["title_ru"].str.contains("philips").map({True: 1, False: 0}).astype(np.uint8)

        # df["num_title_Exclamation"] = df["title"].apply(lambda x: count(x, "!")).astype(np.int16)
        # df["num_title_Question"] = df["title"].apply(lambda x: count(x, "?")).astype(np.int16)
        df["num_desc_Exclamation"] = df["description"].apply(lambda x: count(x, "!")).astype(np.int16)
        df["num_desc_Question"] = df["description"].apply(lambda x: count(x, "?")).astype(np.int16)

    def Do_Drop(df):
        df.drop(["activation_date", "item_id"], axis=1, inplace=True)

    def Do_Stat_Text(df):
        print("feature engineering -> statistics in text ...")
        textfeats = ["text_feature", "text_feature_2", "description", "title"]
        for col in textfeats:
            df[col + "_num_chars"] = df[col].apply(len)
            df[col + "_num_words"] = df[col].apply(lambda comment: len(comment.split()))
            df[col + "_num_unique_words"] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
            df[col + "_words_vs_unique"] = df[col + "_num_unique_words"] / df[col + "_num_words"] * 100
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
    russian_stop1 = set(stopwords.words("russian"))
    russian_stop2 = read_stopwords()
    russian_stop=list(set(russian_stop1).intersection(set(russian_stop2)))
    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": "word",
        "token_pattern": r"\w{1,}",
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": "l2",
        # "min_df":5,
        # "max_df":.9,
        "smooth_idf": False
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
        # ("description", TfidfVectorizer(
        #     ngram_range=(1, 2),
        #     max_features=40000,  # 40000,18000
        #     **tfidf_para,
        #     preprocessor=get_col("description"))
        #  ),
        #         ("title_description", TfidfVectorizer(
        #              ngram_range=(1, 2),#(1,2)
        #              max_features=1800,#40000,18000
        #              **tfidf_para,
        #              preprocessor=get_col("title_description"))
        #           ),
        # ("text_feature", CountVectorizer(
        #     ngram_range=(1, 2),
        #     preprocessor=get_col("text_feature"))
        #  ),

        # ("title", TfidfVectorizer(
        #     ngram_range=(1, 2),
        #     **tfidf_para,
        #     preprocessor=get_col("title"))
        #  ),
        # 新加入两个文本处理title2，title_char
        ("title2", TfidfVectorizer(
            ngram_range=(1, 1),
            **tfidf_para,
            preprocessor=get_col("title"))
         ),

        # ("title", TfidfVectorizer(
        #     ngram_range=(1, 2),
        #     **tfidf_para,
        #     preprocessor=get_col("title_ru"))
        #  ),
        # # 新加入两个文本处理title2，title_char
        # ("title2", TfidfVectorizer(
        #     ngram_range=(1, 1),
        #     **tfidf_para,
        #     preprocessor=get_col("title_ru"))
        #  ),

        #增加了1倍的运行时间
        # ("title_char", TfidfVectorizer(
        #
        #     ngram_range=(1, 4),  # (1, 4),(1,6)
        #     max_features=16000,  # 16000
        #     **tfidf_para2,
        #     preprocessor=get_col("title"))
        #  ),

        # # 新加2018-6-3,速度很慢
        # ("description_feature", CountVectorizer(
        #     ngram_range=(1, 2),
        #     stop_words= russian_stop,
        #  max_features=8000,
        #     preprocessor=get_col("description"))
        #  ),
    ])
    vectorizer.fit(df.to_dict("records"))
    ready_full_df = vectorizer.transform(df.to_dict("records"))
    tfvocab = vectorizer.get_feature_names()
    df.drop(["text_feature", "text_feature_2", "description", "title",
             "title_ru",
             # "title_description"
             ], axis=1, inplace=True)
    df.fillna(-1, inplace=True)
    return df, ready_full_df, tfvocab


# =============================================================================
# Ridge feature https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
# =============================================================================
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if (seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


NFOLDS = 10  # 5
SEED = 42


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

kf = KFold(len_train, n_folds=NFOLDS, shuffle=True, random_state=SEED)


# =============================================================================
# handle price
# =============================================================================
def feature_Eng_On_Price_SEQ(df):
    print('feature engineering -> on price and SEQ ...')
    df["price"] = np.log(df["price"] + 0.001).astype("float32")
    df["price"].fillna(-1, inplace=True)
    df["price+"] = np.round(df["price"] * 2.8).astype(np.int16)  # 4.8
    df["item_seq_number+"] = np.round(df["item_seq_number"] / 100).astype(np.int16)
    return df


train_df, val_df = train_test_split(
    full_df.iloc[:len_train], test_size=0.1, random_state=42)  # 23


def feature_Eng_On_Deal_Prob(df, df_train):
    print('feature engineering -> on price deal prob +...')
    df2 = df
    #    tmp = df_train.groupby(["price+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_price+'})
    #    df = pd.merge(df, tmp, how='left', on=["price+"])
    #    df2['median_deal_probability_price+'] = df['median_deal_probability_price+']
    #    df2['median_deal_probability_price+'] =df2['median_deal_probability_price+'].astype(np.float32)
    #    del tmp; gc.collect()
    #
    #    tmp = df_train.groupby(["item_seq_number+"], as_index=False)['deal_probability'].median().rename(columns={'deal_probability':'median_deal_probability_item_seq_number+'})
    #    df = pd.merge(df, tmp, how='left', on=["item_seq_number+"])
    #    df2['median_deal_probability_item_seq_number+'] = df['median_deal_probability_item_seq_number+']
    #    df2['median_deal_probability_item_seq_number+'] =df2['median_deal_probability_item_seq_number+'].astype(np.float32)

    #    tmp = df.groupby(["image_top_1"], as_index=False)['price'].median().rename(columns={'price':'median_price_image_top_1'})
    #    df = pd.merge(df, tmp, how='left', on=["image_top_1"])
    #    df2['median_price_image_top_1'] = df['median_price_image_top_1']
    #    df2['median_price_image_top_1'] = df2['median_price_image_top_1'].astype(np.float32)
    #    df2['median_price_image_top_1'] = df2['median_price_image_top_1']
    #    df2.fillna(-1, inplace=True)

    #    del tmp; gc.collect()

    return df2


del full_df['deal_probability'];
gc.collect()

# =============================================================================
# use additianl image data
# =============================================================================
feature_engineering(full_df)

feature_Eng_On_Price_SEQ(full_df)
feature_Eng_On_Price_SEQ(train_df)
# 不考虑使用均值
# feature_Eng_On_Deal_Prob(full_df, train_df)

del train_df, test_df;
gc.collect()
full_df, ready_full_df, tfvocab = data_vectorize(full_df)

# 'alpha':20.0
ridge_params = {'alpha': 20.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ready_df = ready_full_df

print('ridge 1 oof ...')
ridge_oof_train, ridge_oof_test = get_oof(ridge, np.array(full_df)[:len_train], y, np.array(full_df)[len_train:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
full_df['ridge_preds_1'] = ridge_preds.astype(np.float32)
full_df['ridge_preds_1'].clip(0.0, 1.0, inplace=True)

ridge_params = {'alpha': 25.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED+1}
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
print('ridge 2 oof ...')
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
full_df['ridge_preds_2'] = ridge_preds.astype(np.float32)
full_df['ridge_preds_2'].clip(0.0, 1.0, inplace=True)

ridge_params = {'alpha': 12.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED+2}
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
print('ridge 3 oof ...')
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:len_train], y, ready_df[len_train:])
ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])
full_df['ridge_preds_3'] = ridge_preds.astype(np.float32)
full_df['ridge_preds_3'].clip(0.0, 1.0, inplace=True)

#read blending.csv
# model_three_df=pd.read_csv('mercari_no2_sol_emb_40_price_ridge_avgday_avgtime_nui_deslen_population_income_512_64_lr1_hpt0_nn_5fold_test.csv')
# full_df['ridge_preds_3'] = model_three_df['deal_probability'].astype(np.float32)
# full_df['ridge_preds_3'].clip(0.0, 1.0, inplace=True)

del ridge_oof_train, ridge_oof_test, ridge_preds, ridge, ready_df
gc.collect()

print("Modeling Stage ...")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(full_df.iloc[:len_train]), ready_full_df[:len_train]])  # Sparse Matrix
tfvocab = full_df.columns.tolist() + tfvocab
X_test_full = full_df.iloc[len_train:]
X_test_ready = ready_full_df[len_train:]
del ready_full_df, full_df
gc.collect()

print("Feature Names Length: ", len(tfvocab))

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

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=42, shuffle=True)
numIter = 0
rmse_sume = 0.
numLimit = 5

for train_index, valid_index in kf.split(y):
    numIter += 1

    if numIter >= numLimit + 1:
        pass
    else:

        print("Modeling Stage ...")

        X_train, X_valid = X.tocsr()[train_index], X.tocsr()[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        gc.collect()
        lgbm_params = {
            "tree_method": "feature",
            "num_threads": 7,
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            # "max_depth": 15,
            "num_leaves": 500,  # 280,360,500,32
            "feature_fraction": 0.2,  # 0.4
            "bagging_fraction": 0.2,  # 0.4
            "learning_rate": 0.015,  # 0.015
            "verbose": -1,
            'lambda_l1': 1,
            'lambda_l2': 1,
            "max_bin": 200,
        }

        lgtrain = lgb.Dataset(X_train, y_train,
                              feature_name=tfvocab,
                              categorical_feature=cat_col)
        lgvalid = lgb.Dataset(X_valid, y_valid,
                              feature_name=tfvocab,
                              categorical_feature=cat_col)
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=32000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=200,
            verbose_eval=100,  # 200
        )

        print("save model ...")
        joblib.dump(lgb_clf, "lgb_{}.pkl".format(numIter))
        ## load model
        # lgb_clf = joblib.load("lgb.pkl")

        print("Model Evaluation Stage")
        print("RMSE:", rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration)))

        test = hstack([csr_matrix(X_test_full), X_test_ready])  # Sparse Matrix
        lgpred = lgb_clf.predict(test, num_iteration=lgb_clf.best_iteration)

        lgsub = pd.DataFrame(lgpred, columns=["deal_probability"], index=sub_item_id)
        lgsub["deal_probability"].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
        lgsub.to_csv("ml_lgb_sub_{}.csv".format(numIter), index=True, header=True)

        rmse_sume += rmse(y_valid, lgb_clf.predict(X_valid, num_iteration=lgb_clf.best_iteration))

        del X_train, X_valid, y_train, y_valid, lgtrain, lgvalid
        gc.collect()

print("mean rmse is:", rmse_sume / numLimit)

print("Features importance...")
bst = lgb_clf
gain = bst.feature_importance("gain")
ft = pd.DataFrame({"feature": bst.feature_name(), "split": bst.feature_importance("split"),
                   "gain": 100 * gain / gain.sum()}).sort_values("gain", ascending=False)
print(ft.head(50))
#
# plt.figure()
# ft[["feature","gain"]].head(50).plot(kind="barh", x="feature", y="gain", legend=False, figsize=(10, 20))
# plt.gcf().savefig("features_importance.png")
print("time costs:{}".format(time.time()-start_time))
print("All Done.")

"""
20w title_ru的新特征
calculating RMSE ...
mean rmse is: 0.225012579055

3个ridge特征
calculating RMSE ...
mean rmse is: 0.22493562355

10w mengfei的特征
calculating RMSE ...
mean rmse is: 0.227470908737

calculating RMSE ...
mean rmse is: 0.227452125002

10w 仅仅统计iPhone等3个特征
calculating RMSE ...
mean rmse is: 0.227380797381

停用词的并集
calculating RMSE ...
mean rmse is: 0.227500445884

停用词的交集
calculating RMSE ...
mean rmse is: 0.227282540687

calculating RMSE ...
mean rmse is: 0.227365660403
1w去掉char功能
calculating RMSE ...
mean rmse is: 0.2358583443750936
time costs:206.2117302417755

1w 不加新特征
calculating RMSE ...
mean rmse is: 0.235880369348843

mean rmse is: 0.2358012922087179

1w 新特征 5折
calculating RMSE ...
mean rmse is: 0.23640032088836488

calculating RMSE ...
mean rmse is: 0.236380033591672

calculating RMSE ...
mean rmse is: 0.23623534673154473

5w 不要新特征
calculating RMSE ...
mean rmse is: 0.23150961548765264

5w 要新特征
calculating RMSE ...
mean rmse is: 0.23133951850101137

10w 原来 5折平均
calculating RMSE ...
mean rmse is: 0.2275115776789228

改进后 5折平均
calculating RMSE ...
mean rmse is: 0.22747084242905377

mean rmse is: 0.22719179805910664
time costs:4384.6224999427795

calculating RMSE ...
mean rmse is: 0.22723049318423594
time costs:4685.339641332626

calculating RMSE ...
mean rmse is: 0.22724368413992296
time costs:3650.90975856781

10w
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 0.217028	valid's rmse: 0.234943
[200]	train's rmse: 0.192643	valid's rmse: 0.233402
[300]	train's rmse: 0.173163	valid's rmse: 0.233263
[400]	train's rmse: 0.157665	valid's rmse: 0.234255
Early stopping, best iteration is:
[239]	train's rmse: 0.18445	valid's rmse: 0.23315
save model ...
Model Evaluation Stage
calculating RMSE ...
RMSE: 0.23315031644628909
/home/deepcam/anaconda2/envs/py36/lib/python3.6/site-packages/lightgbm/basic.py:447: UserWarning: Converting data to scipy sparse matrix.
  warnings.warn('Converting data to scipy sparse matrix.')
calculating RMSE ...
Modeling Stage ...
[LightGBM] [Warning] Unknown parameter: tree_method
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 0.215289	valid's rmse: 0.247115
[200]	train's rmse: 0.191162	valid's rmse: 0.246157
[300]	train's rmse: 0.171891	valid's rmse: 0.246581
Early stopping, best iteration is:
[162]	train's rmse: 0.199329	valid's rmse: 0.246073
save model ...
Model Evaluation Stage
calculating RMSE ...
RMSE: 0.24607253175351204
calculating RMSE ...
Modeling Stage ...
[LightGBM] [Warning] Unknown parameter: tree_method
[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
[LightGBM] [Warning] Met negative value in categorical features, will convert it to NaN
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 0.215796	valid's rmse: 0.240517
[200]	train's rmse: 0.19131	valid's rmse: 0.238724
[300]	train's rmse: 0.172705	valid's rmse: 0.239144
[400]	train's rmse: 0.157289	valid's rmse: 0.239466
Early stopping, best iteration is:
[208]	train's rmse: 0.189765	valid's rmse: 0.238603
save model ...
Model Evaluation Stage
calculating RMSE ...
RMSE: 0.2386025086637869
calculating RMSE ...
Modeling Stage ...
[LightGBM] [Warning] Unknown parameter: tree_method
Training until validation scores don't improve for 200 rounds.
[100]	train's rmse: 0.216788	valid's rmse: 0.234751
[200]	train's rmse: 0.191797	valid's rmse: 0.231529
[300]	train's rmse: 0.172888	valid's rmse: 0.230917
[400]	train's rmse: 0.157764	valid's rmse: 0.230966
[500]	train's rmse: 0.145616	valid's rmse: 0.230711
[600]	train's rmse: 0.134631	valid's rmse: 0.230809
Early stopping, best iteration is:
[487]	train's rmse: 0.147029	valid's rmse: 0.230622



full data
fold 1: [2787]  train's rmse: 0.17505   valid's rmse: 0.215004
fold 2: [2586]  train's rmse: 0.176207  valid's rmse: 0.214477
100k data
-------------------------------------------------------------------------------------------
mean rmse is: 0.22698800710415235 - reduce max_bin to 200 - 2folds LB:0.2207
mean rmse is: 0.22710252553320615 - add median_price_image_top_1 #在2208版去除
mean rmse is: 0.22723174589672093 - add image information ---------------------------------
mean rmse is: 0.22736661491719415 - add price_vs_city_income #在2fold-2207版本去除
mean rmse is: 0.22768780227534266 - price+ 4.8 to 2.8 -------------------------------------
mean rmse is: 0.22780198548953648 - original - 2folds LB 0.2198
[200]   train's rmse: 0.216075  valid's rmse: 0.220928
[300]   train's rmse: 0.211498  valid's rmse: 0.218822
[400]   train's rmse: 0.208222  valid's rmse: 0.217763
[500]   train's rmse: 0.205433  valid's rmse: 0.217029
[600]   train's rmse: 0.202888  valid's rmse: 0.216503
[700]   train's rmse: 0.200629  valid's rmse: 0.216155
[800]   train's rmse: 0.198475  valid's rmse: 0.215904
[900]   train's rmse: 0.196621  valid's rmse: 0.215725
[1000]  train's rmse: 0.194879  valid's rmse: 0.215554
[1100]  train's rmse: 0.193265  valid's rmse: 0.215432
[1200]  train's rmse: 0.191686  valid's rmse: 0.215331
[1300]  train's rmse: 0.19016   valid's rmse: 0.215254
[1400]  train's rmse: 0.188807  valid's rmse: 0.215205
[1500]  train's rmse: 0.187473  valid's rmse: 0.215139
[1600]  train's rmse: 0.18622   valid's rmse: 0.215099
[1700]  train's rmse: 0.185041  valid's rmse: 0.215064
[1800]  train's rmse: 0.183911  valid's rmse: 0.21503
[1900]  train's rmse: 0.182841  valid's rmse: 0.215007
[2000]  train's rmse: 0.181771  valid's rmse: 0.214982
[2100]  train's rmse: 0.180755  valid's rmse: 0.21495
[2200]  train's rmse: 0.179723  valid's rmse: 0.214936
[2300]  train's rmse: 0.178825  valid's rmse: 0.21492
[2400]  train's rmse: 0.177879  valid's rmse: 0.214896
[2500]  train's rmse: 0.176977  valid's rmse: 0.21488
[2600]  train's rmse: 0.176063  valid's rmse: 0.214868
[2700]  train's rmse: 0.17521   valid's rmse: 0.214865
"""