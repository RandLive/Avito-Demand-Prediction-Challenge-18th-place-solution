#Based on https://www.kaggle.com/tezdhar/wordbatch-with-memory-test
import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import psutil
import os

import wordbatch

from wordbatch.extractors import WordBag
from wordbatch.models import FTRL, FM_FTRL

from nltk.corpus import stopwords
import re


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))




def handle_missing_inplace(dataset):
    dataset['description'].fillna(value='na', inplace=True)
    dataset["image"].fillna("noinformation", inplace=True)
    dataset["param_1"].fillna("nicapotato", inplace=True)
    dataset["param_2"].fillna("nicapotato", inplace=True)
    dataset["param_3"].fillna("nicapotato", inplace=True)

    dataset['image_top_1'].fillna(value=-1, inplace=True)
    dataset['price'].fillna(value=0, inplace=True)






def to_categorical(dataset):

    dataset['param_1'] = dataset['param_1'].astype('category')
    dataset['param_2'] = dataset['param_2'].astype('category')
    dataset['param_3'] = dataset['param_3'].astype('category')

    dataset['image_top_1'] = dataset['image_top_1'].astype('category')
    dataset['image'] = dataset['image'].astype('category')
    dataset['price'] = dataset['price'].astype('category')
    #counting
    dataset['num_desc_punct'] = dataset['num_desc_punct'].astype('category')
    dataset['num_desc_capE'] = dataset['num_desc_capE'].astype('category')
    dataset['num_desc_capP'] = dataset['num_desc_capP'].astype('category')
    dataset['num_title_punct'] = dataset['num_title_punct'].astype('category')
    dataset['num_title_capE'] = dataset['num_title_capE'].astype('category')
    dataset['num_title_capP'] = dataset['num_title_capP'].astype('category')
    dataset['is_in_desc_хорошо'] = dataset['is_in_desc_хорошо'].astype('category')
    dataset['is_in_desc_Плохо'] = dataset['is_in_desc_Плохо'].astype('category')
    dataset['is_in_desc_новый'] = dataset['is_in_desc_новый'].astype('category')
    dataset['is_in_desc_старый'] = dataset['is_in_desc_старый'].astype('category')
    dataset['is_in_desc_используемый'] = dataset['is_in_desc_используемый'].astype('category')
    dataset['is_in_desc_есплатная_доставка'] = dataset['is_in_desc_есплатная_доставка'].astype('category')
    dataset['is_in_desc_есплатный_возврат'] = dataset['is_in_desc_есплатный_возврат'].astype('category')
    dataset['is_in_desc_идеально'] = dataset['is_in_desc_идеально'].astype('category')
    dataset['is_in_desc_подержанный'] = dataset['is_in_desc_подержанный'].astype('category')
    dataset['is_in_desc_пСниженные_цены'] = dataset['is_in_desc_пСниженные_цены'].astype('category')

    #region
    dataset['region'] = dataset['region'].astype('category')
    dataset['city'] = dataset['city'].astype('category')
    dataset['user_type'] = dataset['user_type'].astype('category')
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['parent_category_name'] = dataset['parent_category_name'].astype('category')


    # dataset['price+'] = dataset['price+'].astype('category')

    # dataset['desc_len'] = dataset['desc_len'].astype('category')
    # dataset['title_len'] = dataset['title_len'].astype('category')
    # dataset['title_desc_len_ratio'] = dataset['title_desc_len_ratio'].astype('category')
    # dataset['desc_word_count'] = dataset['desc_word_count'].astype('category')
    # dataset['mean_des'] = dataset['mean_des'].astype('category')




# Define helpers for text normalization
stopwords = {x: 1 for x in stopwords.words('russian')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')


def normalize_text(text):
    # if np.isnan(text): text='na'
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])



develop = True
# develop= False
if __name__ == '__main__':
    start_time = time.time()
    from time import gmtime, strftime
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    import  re
    from scipy.sparse import hstack
    from nltk.corpus import stopwords
    from contextlib import contextmanager
    @contextmanager
    def timer(name):
        t0 = time.time()
        yield
        print('[{}] done in {:.0f} s'.format(name, (time.time() - t0)))
    import string
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    print("\nData Load Stage")
    # , nrows = nrows
    nrows=10000*1
    training = pd.read_csv('../input/train.csv', index_col="item_id", parse_dates=["activation_date"])
    len_train = len(training)
    traindex = training.index
    testing = pd.read_csv('../input/test.csv',  index_col="item_id", parse_dates=["activation_date"])
    testdex = testing.index
    # labels = training['deal_probability'].values
    y = training.deal_probability.copy()
    training.drop("deal_probability", axis=1, inplace=True)
    # suppl
    # used_cols = ["item_id", "user_id"]
    # train_active = pd.read_csv("../input/train_active.csv", usecols=used_cols)
    # test_active = pd.read_csv("../input/test_active.csv", usecols=used_cols)
    # train_periods = pd.read_csv("../input/periods_train.csv", parse_dates=["date_from", "date_to"])
    # test_periods = pd.read_csv("../input/periods_test.csv", parse_dates=["date_from", "date_to"])

    # =============================================================================
    # Add region-income
    # =============================================================================
    tmp = pd.read_csv("../input/region_income.csv",  sep=";", names=["region", "income"])
    training = training.merge(tmp, on="region", how="left")
    testing = testing.merge(tmp, on="region", how="left")
    del tmp;
    gc.collect()
    # =============================================================================
    # Add region-income
    # =============================================================================
    tmp = pd.read_csv("../input/city_population_wiki_v3.csv",)
    training = training.merge(tmp, on="city", how="left")
    testing = testing.merge(tmp, on="city", how="left")
    del tmp;
    gc.collect()
    import pickle
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
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

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
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

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
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding average_pixel_width ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_average_pixel_width = x['average_pixel_width']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_average_pixel_width = x['average_pixel_width']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_average_pixel_width, columns=['average_pixel_width'])
    incep_test_image_df = pd.DataFrame(test_average_pixel_width, columns=[f'average_pixel_width'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding average_reds ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_average_reds = x['average_reds']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_average_reds = x['average_reds']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_average_reds, columns=['average_reds'])
    incep_test_image_df = pd.DataFrame(test_average_reds, columns=[f'average_reds'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding average_blues ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_average_blues = x['average_blues']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_average_blues = x['average_blues']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_average_blues, columns=['average_blues'])
    incep_test_image_df = pd.DataFrame(test_average_blues, columns=[f'average_blues'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding average_greens ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_average_greens = x['average_greens']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_average_greens = x['average_greens']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_average_greens, columns=['average_greens'])
    incep_test_image_df = pd.DataFrame(test_average_greens, columns=[f'average_greens'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding widths ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_widths = x['widths']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_widths = x['widths']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_widths, columns=['widths'])
    incep_test_image_df = pd.DataFrame(test_widths, columns=[f'widths'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    print('adding heights ...')
    with open('../input/train_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    train_heights = x['heights']
    train_ids = x['ids']

    with open('../input/test_image_features_1.p', 'rb') as f:
        x = pickle.load(f)

    test_heights = x['heights']
    test_ids = x['ids']
    del x;
    gc.collect()

    incep_train_image_df = pd.DataFrame(train_heights, columns=['heights'])
    incep_test_image_df = pd.DataFrame(test_heights, columns=[f'heights'])
    incep_train_image_df['image'] = (train_ids)
    incep_test_image_df['image'] = (test_ids)
    training = training.join(incep_train_image_df.set_index('image'), on='image')
    testing = testing.join(incep_test_image_df.set_index('image'), on='image')

    # ==============================================================================
    # image features by Qifeng
    # ==============================================================================
    print('adding image features ...')
    with open('../input/train_image_features_cspace.p', 'rb') as f:
        x = pickle.load(f)

    x_train = pd.DataFrame(x, columns=['average_HSV_Ss', \
                                       'average_HSV_Vs', \
                                       'average_LUV_Ls', \
                                       'average_LUV_Us', \
                                       'average_LUV_Vs', \
                                       'average_HLS_Hs', \
                                       'average_HLS_Ls', \
                                       'average_HLS_Ss', \
                                       'average_YUV_Ys', \
                                       'average_YUV_Us', \
                                       'average_YUV_Vs', \
                                       'ids'
                                       ])
    # x_train.rename(columns = {'$ids':'image'}, inplace = True)

    with open('../input/test_image_features_cspace.p', 'rb') as f:
        x = pickle.load(f)

    x_test = pd.DataFrame(x, columns=['average_HSV_Ss', \
                                      'average_HSV_Vs', \
                                      'average_LUV_Ls', \
                                      'average_LUV_Us', \
                                      'average_LUV_Vs', \
                                      'average_HLS_Hs', \
                                      'average_HLS_Ls', \
                                      'average_HLS_Ss', \
                                      'average_YUV_Ys', \
                                      'average_YUV_Us', \
                                      'average_YUV_Vs', \
                                      'ids'
                                      ])
    # x_test.rename(columns = {'$ids':'image'}, inplace = True)

    training = training.join(x_train.set_index('ids'), on='image')
    testing = testing.join(x_test.set_index('ids'), on='image')
    del x, x_train, x_test;
    gc.collect()

    print('Train shape: {} Rows, {} Columns'.format(*training.shape))
    print('Test shape: {} Rows, {} Columns'.format(*testing.shape))
    # Combine Train and Test
    merge = pd.concat([training, testing], axis=0)
    trnshape = training.shape[0]

    gc.collect()

    # handle_missing_inplace(test)

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))
    # some count
    count = lambda l1, l2: sum([1 for x in l1 if x in l2])
    merge["text_feature"] = merge.apply(lambda row: " ".join([str(row["param_1"]),
                                                        str(row["param_2"]), str(row["param_3"])]), axis=1)

    merge["num_desc_punct"] = merge["description"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
    merge["num_desc_capE"] = merge["description"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
    merge["num_desc_capP"] = merge["description"].apply(lambda x: count(x, "[А-Я]")).astype(np.int16)

    merge["num_title_punct"] = merge["title"].apply(lambda x: count(x, set(string.punctuation))).astype(np.int16)
    merge["num_title_capE"] = merge["title"].apply(lambda x: count(x, "[A-Z]")).astype(np.int16)
    merge["num_title_capP"] = merge["title"].apply(lambda x: count(x, "[А-Я]")).astype(np.int16)
    # good, used, bad ... count
    merge["is_in_desc_хорошо"] = merge["description"].str.contains("хорошо").map({True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_Плохо"] = merge["description"].str.contains("Плохо").map({True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_новый"] = merge["description"].str.contains("новый").map({True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_старый"] = merge["description"].str.contains("старый").map({True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_используемый"] = merge["description"].str.contains("используемый").map(
        {True: 1, False: 0}).astype(
        np.uint8)
    merge["is_in_desc_есплатная_доставка"] = merge["description"].str.contains("есплатная доставка").map(
        {True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_есплатный_возврат"] = merge["description"].str.contains("есплатный возврат").map(
        {True: 1, False: 0}).astype(np.uint8)
    merge["is_in_desc_идеально"] = merge["description"].str.contains("идеально").map({True: 1, False: 0}).astype(
        np.uint8)
    merge["is_in_desc_подержанный"] = merge["description"].str.contains("подержанный").map({True: 1, False: 0}).astype(
        np.uint8)
    merge["is_in_desc_пСниженные_цены"] = merge["description"].str.contains("Сниженные цены").map(
        {True: 1, False: 0}).astype(np.uint8)

    #new features
    # merge['desc_len'] = np.log1p(merge['description'].apply(lambda x: len(x)))
    # merge['title_len'] = np.log1p(merge['title'].apply(lambda x: len(x)))
    # merge['title_desc_len_ratio'] = np.log1p(merge['title_len'] / merge['desc_len'])
    # #
    # merge['desc_word_count'] = merge['description'].apply(lambda x: len(x.split()))
    # merge['mean_des'] = merge['description'].apply(
    #     lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
    # merge['title_word_count'] = merge['title'].apply(lambda x: len(x.split()))
    # merge['mean_title'] = merge['title'].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10

    # merge.drop('category_name', axis=1, inplace=True)
    # merge["price+"] = np.round(merge["price"] * 2.8).astype(np.int16)  # 4.8
    # merge["item_seq_number+"] = np.round(merge["item_seq_number"] / 100).astype(np.int16)
    merge["item_seq_number"]=np.log1p(merge["item_seq_number"]).astype(np.float32)
    merge["price"].fillna(-1, inplace=True)
    # merge["wday"] = merge["activation_date"].dt.weekday
    # merge["wday"] = merge["wday"].astype('category')
    # merge["is_in_title_iphone"] = merge["title"].str.contains("iphone").map({True: 1, False: 0}).astype('category')
    # merge["is_in_desc_ipod"] = merge["title"].str.contains("ipod").map({True: 1, False: 0}).astype('category')
    #
    # merge["is_in_desc_samsung"] = merge["title"].str.contains("samsung").map({True: 1, False: 0}).astype('category')
    merge['blurinesses']=merge['blurinesses'].astype('category')
    merge['whitenesses']=merge['whitenesses'].astype('category')
    merge['dullnesses']=merge['dullnesses'].astype('category')
    merge['average_pixel_width']=merge['average_pixel_width'].astype('category')
    merge['average_reds'] = merge['average_reds'].astype('category')

    merge['average_blues'] = merge['average_blues'].astype('category')
    merge['average_greens'] = merge['average_greens'].astype('category')
    merge['widths'] = merge['widths'].astype('category')
    merge['heights'] = merge['heights'].astype('category')

    merge['average_HSV_Ss'] = merge['average_HSV_Ss'].astype('category')
    merge['average_HSV_Vs'] = merge['average_HSV_Vs'].astype('category')
    merge['average_LUV_Ls'] = merge['average_LUV_Ls'].astype('category')
    merge['average_LUV_Us'] = merge['average_LUV_Us'].astype('category')
    merge['average_LUV_Vs'] = merge['average_LUV_Vs'].astype('category')
    merge['average_HLS_Hs'] = merge['average_HLS_Hs'].astype('category')
    merge['average_HLS_Ls'] = merge['average_HLS_Ls'].astype('category')
    merge['average_HLS_Ss'] = merge['average_HLS_Ss'].astype('category')
    merge['average_YUV_Ys'] = merge['average_YUV_Ys'].astype('category')
    merge['average_YUV_Us'] = merge['average_YUV_Us'].astype('category')
    merge['average_YUV_Vs'] = merge['average_YUV_Vs'].astype('category')


    # merge["price+"] = np.round(merge["price"] * 2.8).astype(np.int16)  # 4.8
    print('[{}] Do some counting completed.'.format(time.time() - start_time))



    with timer("Log features"):
        for fea in ['price','image_top_1']:
            merge[fea]= np.log2(1 + merge[fea].values).astype(int)


    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))


    def text_preprocessing(text):
        # text = str(text)
        # text = text.lower()
        # text = re.sub(r"(\\u[0-9A-Fa-f]+)", r"", text)
        # text = re.sub(r"===", r" ", text)
        # # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
        # text = " ".join(map(str.strip, re.split('(\d+)', text)))
        # regex = re.compile(u'[^[:alpha:]]')
        # text = regex.sub(" ", text)
        # text = " ".join(text.split())
        return text


    merge['description']=merge['description'].apply(text_preprocessing)
    merge['title'] = merge['title'].apply(text_preprocessing)
    wb = wordbatch.WordBatch( extractor=(WordBag, {"hash_ngrams": 1, "hash_ngrams_weights": [1.5, 1.0],
                                                                  "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                                  "idf": None,
                                                                  }), procs=16)

    wb.dictionary_freeze = True
    # wb.fit(train['name'])
    min_df_name = 5#5，3
    X_desc_merge = wb.fit_transform(merge['description'])
    X_title_merge = wb.transform(merge['title'])
    del (wb)
    mask_name = np.where(X_desc_merge.getnnz(axis=0) > min_df_name)[0]
    X_desc_merge = X_desc_merge[:, mask_name]

    mask_name = np.where(X_title_merge.getnnz(axis=0) > min_df_name)[0]
    X_title_merge = X_title_merge[:, mask_name]


    print('[{}] Vectorize `desc,title` completed.'.format(time.time() - start_time))

    wb = CountVectorizer()

    X_param_1 = wb.fit_transform(merge['param_1'])
    X_param_2 = wb.fit_transform(merge['param_2'])
    X_param_3 = wb.fit_transform(merge['param_3'])



    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    # lb = LabelBinarizer(sparse_output=True)
    # X_brand = lb.fit_transform(merge['brand_name'])
    # X_brand_train = X_brand[:nrow_train]
    # X_brand_test = X_brand[nrow_test:]
    # del X_brand
    # print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['param_1', 'param_2', 'param_3',
                                                 'image_top_1',
                                                 # 'region'
                                                 'is_in_desc_хорошо', 'is_in_desc_Плохо',
                                                 'is_in_desc_новый', 'is_in_desc_старый', 'is_in_desc_используемый',
                                                 'is_in_desc_есплатная_доставка',
                                                 'is_in_desc_есплатный_возврат', 'is_in_desc_идеально',
                                                 'is_in_desc_подержанный', 'is_in_desc_пСниженные_цены',
                                                 # 'price+',
                                                 # 'city',
                                                 'user_type',
                                                 'category_name',
                                                 "parent_category_name",
                                                 # "wday"
                                                 # "is_in_title_iphone","is_in_desc_ipod","is_in_desc_samsung",
                                                 ]],
                                          sparse=True).values)


    # X_dummies_test = X_dummies[nrow_test:]
    # del X_dummies
    print('[{}] Get dummies on `param_1` and `param_2,param_3` completed.'.format(time.time() - start_time))
    #
    X_counting = csr_matrix(pd.get_dummies(merge[
                                               ['num_desc_punct', 'num_desc_capE', 'num_desc_capP', 'num_title_punct',
                                                'num_title_capP', 'num_title_capE',
                                                # 'is_in_desc_хорошо', 'is_in_desc_Плохо',
                                                # 'is_in_desc_новый', 'is_in_desc_старый', 'is_in_desc_используемый',
                                                # 'is_in_desc_есплатная_доставка',
                                                # 'is_in_desc_есплатный_возврат', 'is_in_desc_идеально',
                                                # 'is_in_desc_подержанный', 'is_in_desc_пСниженные_цены',
                                                'price',
                                                # 'desc_len','title_len',
                                                # 'title_desc_len_ratio',
                                                # 'desc_word_count',
                                                # 'mean_des',
                                                # 'title_word_count',
                                                # 'mean_title'
                                                # 'price+',
                                                "item_seq_number",
                                                'blurinesses','whitenesses','dullnesses',
                                                'average_pixel_width','average_reds',
                                                'average_blues', 'average_greens', 'widths', 'heights',
                                                'average_HSV_Ss', \
                                                'average_HSV_Vs', \
                                                'average_LUV_Ls', \
                                                'average_LUV_Us', \
                                                'average_LUV_Vs', \
                                                'average_HLS_Hs', \
                                                'average_HLS_Ls', \
                                                'average_HLS_Ss', \
                                                'average_YUV_Ys', \
                                                'average_YUV_Us', \
                                                'average_YUV_Vs', \
 \
                                                ]
                                           ],
                                           sparse=True).values)
    # X_counting_train=X_counting[:nrow_train]
    # X_counting_test = X_counting[nrow_test:]
    # del X_counting
    # if develop:
    #     print(u'memory：{}gb'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #
    # wb = CountVectorizer()
    # merge['catship'] = merge['category_name'].astype(str) + merge['shipping'].astype(str)
    # X_catship = wb.fit_transform(merge['catship'])
    # X_catship_train = X_catship[:nrow_train]
    # X_catship_test = X_catship[nrow_test:]
    # del merge
    # print('[{}] Count vectorize `catship` completed.'.format(time.time() - start_time))
    # if develop:
    #     print(u'memory：{}gb'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #
    # print(X_dummies_train.shape, X_description_train.shape, X_brand_train.shape,
    #       X_category1_train.shape, X_category2_train.shape, X_category3_train.shape,
    #       X_name_train.shape, X_category_train.shape, X_catship_train.shape)
    #
    from nltk.corpus import stopwords


    def char_analyzer(text):
        """
        This is used to split strings in small lots
        anttip saw this in an article
        so <talk> and <talking> would have <Tal> <alk> in common
        should be similar to russian I guess
        """
        tokens = text.split()
        return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]
    stopWords = stopwords.words('russian')
    with timer("Tfidf on description word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=lambda x: re.findall(r'[^{P}\W]+', x),
            analyzer='word',
            token_pattern=None,
            stop_words=stopWords,
            ngram_range=(1, 4),#1,2
            max_features=200000)
        X = word_vectorizer.fit_transform(merge['description'])
        train_word_features = X[:len_train]
        test_word_features = X[len_train:]
        del (X)
    with timer("Tfidf on title word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=lambda x: re.findall(r'[^{P}\W]+', x),
            analyzer='word',
            token_pattern=None,
            stop_words=stopWords,
            ngram_range=(3, 6),#1,2
            max_features=50000)
        X = word_vectorizer.fit_transform(merge['title'])
        train_title_word_features = X[:trnshape]
        test_title_word_features = X[trnshape:]
        del (X)
    with timer("Tfidf on char n_gram"):
        char_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=char_analyzer,
            analyzer='word',
            ngram_range=(1, 4),#
            max_features=50000)
        X = char_vectorizer.fit_transform(merge['title'])
        train_char_features = X[:trnshape]
        test_char_features = X[trnshape:]
        del (X)
    with timer("Tfidf on description word"):
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            tokenizer=lambda x: re.findall(r'[^{P}\W]+', x),
            analyzer='word',
            token_pattern=None,
            stop_words=stopWords,
            ngram_range=(3, 6),#1,2
            max_features=200000)#200000
        X = word_vectorizer.fit_transform(merge['description'])
        train_desc_word_features = X[:trnshape]
        test_desc_word_features = X[trnshape:]
        del (X)
    # with timer("CountVectorizer on word"):
    #     word_vectorizer = CountVectorizer(
    #         ngram_range=(3, 5),)
    #     X = word_vectorizer.fit_transform(merge['text_feature'])
    #     train_text_feature_word_features = X[:trnshape]
    #     test_text_feature_word_features = X[trnshape:]
    #     del (X)

    X = hstack((X_dummies[:len_train],X_counting[:len_train], X_param_1[:len_train],
                      X_param_2[: len_train],X_param_3[:len_train],X_desc_merge[:len_train],
                      X_title_merge[: len_train],
                      train_word_features,
                      # train_text_feature_word_features,
                      train_char_features,train_title_word_features,
                train_desc_word_features
                      )).tocsr()
    X_test = hstack((X_dummies[len_train:], X_counting[len_train:], X_param_1[len_train:],
                      X_param_2[len_train:], X_param_3[len_train:], X_desc_merge[len_train:],
                      X_title_merge[len_train:],
                     test_word_features,
                     # test_text_feature_word_features,
                     test_char_features,test_title_word_features,
                     test_desc_word_features
                      )).tocsr()
    del X_title_merge,X_counting,X_desc_merge,X_dummies,X_param_1,X_param_2,X_param_3,merge, \
        train_word_features,test_word_features,\
        train_char_features, train_title_word_features,\
        test_char_features, test_title_word_features,
    train_desc_word_features,test_desc_word_features,
    merge
    gc.collect()

    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    from sklearn.model_selection import KFold
    nfold = 5
    kf = KFold(n_splits=nfold, random_state=42, shuffle=True)
    fold_id = -1



    val_predict = np.zeros(y.shape)
    aver_rmse = 0.0
    for train_index, val_index in kf.split(y):
        fold_id += 1
        print("Fold {} start...".format(fold_id))
        train_X, valid_X = X[train_index], X[val_index]
        train_y, valid_y = y[train_index], y[val_index]

    # train_X, valid_X, train_y, valid_y = train_test_split(X_train, y, test_size=0.2, random_state=42)
    # del X_train, y

        d_shape = train_X.shape[1]
        print('d_shape', d_shape)
        model = FTRL(alpha=0.01, beta=0.1, L1=0.1, L2=10, D=d_shape, iters=5, inv_link="identity", threads=8)
        model.fit(train_X, train_y)
        def rmse(predictions, targets):
            print("calculating RMSE ...")
            return np.sqrt(((predictions - targets) ** 2).mean())
        preds_valid_ftrl = model.predict(X=valid_X)
        # print(" FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds_valid_ftrl)))
        print(" FTRL dev RMSLE:", rmse(valid_y, preds_valid_ftrl))
        # # model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X_train.shape[1], alpha_fm=0.01, L2_fm=0.0,
        # #                 init_fm=0.01,
        # #                 D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)
        model = FM_FTRL(alpha=0.01, beta=0.01, L1=1, L2=8, D=d_shape, alpha_fm=0.01, L2_fm=0.0,
                        init_fm=0.01,
                        D_fm=200,  iters=6, inv_link="identity", threads=8)



        model.fit(train_X, train_y)
        preds_valid_fm = model.predict(X=valid_X)
        del valid_X,train_X, train_y
        # print("FM  dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds_valid_fm)))
        print("FM  dev RMSLE:", rmse(valid_y, preds_valid_fm))
        def aggregate_predicts3(Y1, Y2, ratio1):
            assert Y1.shape == Y2.shape
            return Y1 * ratio1 + Y2 * (1-ratio1)

        weight = 0.5#0.5
        preds = weight * preds_valid_fm + (1 - weight) * preds_valid_ftrl
        val_predict[val_index] = preds
        # print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
        print("FM_FTRL dev RMSLE:", rmse(valid_y, preds))
        aver_rmse += rmse(valid_y, preds)
        sub = pd.read_csv('../input/sample_submission.csv', )  # , nrows=10000*5
        pred = model.predict(X_test)
        sub['deal_probability'] = pred
        sub['deal_probability'].clip(0.0, 1.0, inplace=True)
        print("Output Prediction CSV")
        sub.to_csv('subm/ftrl_fm_submissionV3_{}.csv'.format(fold_id), index=False)


    print("average rmse:{}".format(aver_rmse / nfold))
    train_data = pd.read_csv('../input/train.csv', )
    label = ['deal_probability']
    # train_user_ids = train_data.user_id.values
    train_item_ids = train_data.item_id.values

    train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
    # train_user_ids = train_item_ids.reshape(len(train_user_ids), 1)
    val_predicts = pd.DataFrame(data=val_predict, columns=label)
    # val_predicts['user_id'] = train_user_ids
    val_predicts['item_id'] = train_item_ids
    val_predicts.to_csv('subm/ftrl_fmV3_train.csv', index=False)
    # # ratio optimum finder for 3 models
    # best1 = 0
    # lowest = 0.99
    # for i in range(100):
    #     r = i * 0.01
    #     if r  < 1.0:
    #         Y_dev_preds = aggregate_predicts3(preds_valid_fm, preds_valid_ftrl,  r)
    #         fpred = rmsle(np.expm1(valid_y), np.expm1(Y_dev_preds))
    #         if fpred < lowest:
    #             best1 = r
    #             lowest = fpred
    # # print(str(r)+"-RMSL error for RNN + Ridge + RidgeCV on dev set:", fpred)
    # Y_dev_preds = Y_dev_preds = aggregate_predicts3(preds_valid_fm, preds_valid_ftrl,  best1)
    # print(best1)
    #
    # print("(Best) RMSL error for RNN attention + FM + fasttext on dev set:", rmsle(np.expm1(valid_y), np.expm1(Y_dev_preds)))
    #
    # del train_X,train_y
    # print('[{}] Train ridge v2 completed'.format(time.time() - start_time))
    # if develop:
    #     print(u'memory：{}gb'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    #
    # if develop:
    #     preds = model.predict(X=valid_X)
    #     print("FM_FTRL dev RMSLE:", rmsle(np.expm1(valid_y), np.expm1(preds)))
    #
    # predsFM = model.predict(X_test)
    # print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))
    # del X_test
    # gc.collect()
    # preds = predsFM
    #
    # submission['price'] = np.expm1(preds)
    # submission.to_csv("submission_wordbatch_ftrl_fm.csv", index=False)
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print("Total time costs:{}".format(time.time()-start_time))
    '''
    1w oof
    Output Prediction CSV
    average rmse:0.244059242699125

    average rmse:0.2436276297688677
    
    average rmse:0.24356555599510213
    average rmse:0.2435573104461158
    average rmse:0.2433847234729977
    average rmse:0.2430078486780956
    average rmse:0.24299577528880514
    
    average rmse:0.24297421901652375
    average rmse:0.2429121752449718
    
    average rmse:0.24265357451463557
    
    average rmse:0.24252233100353485
    average rmse:0.24247167824739174
    
    average rmse:0.24219114762025212
    
    average rmse:0.24154770841924011
    2018-06-24 08:22:34
    Total time costs:632.5025238990784
    
2018-06-24 04:37:31

    5w oof
    average rmse:0.23784727588111987
    
    average rmse:0.2374566037691721
    
    全数据oof
    average rmse:0.22752099248003815
    2018-06-24 01:34:11


    1w
     FTRL dev RMSLE: 0.236763228999
    FM  dev RMSLE: 0.236599859025
    
    FTRL dev RMSLE: 0.236763228999
FM  dev RMSLE: 0.236259762027
FM_FTRL dev RMSLE: 0.236307818481
    
    加image_top_1
    FTRL dev RMSLE: 0.2365524478
FM  dev RMSLE: 0.236126474398
FM_FTRL dev RMSLE: 0.236139755285

加desc tfidf vec
d_shape 45372
 FTRL dev RMSLE: 0.235671709213
FM  dev RMSLE: 0.236226484101
FM_FTRL dev RMSLE: 0.236117518185

d_shape 45372
 FTRL dev RMSLE: 0.235671709213
FM  dev RMSLE: 0.236159988691
FM_FTRL dev RMSLE: 0.235875096042
2018-06-23 10:32:17

d_shape 221372
 FTRL dev RMSLE: 0.235755131719
FM  dev RMSLE: 0.236065858162
FM_FTRL dev RMSLE: 0.235762389065
2018-06-23 10:40:53



d_shape 264733
calculating RMSE ...
 FTRL dev RMSLE: 0.236015070756
calculating RMSE ...
FM  dev RMSLE: 0.235691015213
calculating RMSE ...
FM_FTRL dev RMSLE: 0.235624786658
2018-06-23 10:53:34

d_shape 264733
calculating RMSE ...
 FTRL dev RMSLE: 0.243936175374
calculating RMSE ...
FM  dev RMSLE: 0.242935442428
calculating RMSE ...
FM_FTRL dev RMSLE: 0.243110294203
2018-06-23 11:00:10

ftrl 选择0.01学习率，fm选择0.012学习率，终于两个结合在一起出现了提升的情况
d_shape 264733
calculating RMSE ...
 FTRL dev RMSLE: 0.242810536001
calculating RMSE ...
FM  dev RMSLE: 0.242935442428
calculating RMSE ...
FM_FTRL dev RMSLE: 0.242692624406
2018-06-23 11:03:27

5w
[55.92116975784302] Create sparse merge completed
d_shape 283708
calculating RMSE ...
 FTRL dev RMSLE: 0.241283110197
calculating RMSE ...
FM  dev RMSLE: 0.241620178685
calculating RMSE ...
FM_FTRL dev RMSLE: 0.241369959742
2018-06-23 11:09:12

去掉文本预处理
d_shape 283276
calculating RMSE ...
 FTRL dev RMSLE: 0.240947392605
calculating RMSE ...
FM  dev RMSLE: 0.24137407715
calculating RMSE ...
FM_FTRL dev RMSLE: 0.241087603203
2018-06-23 11:53:57

 FTRL dev RMSLE: 0.24038598037733158
Total e: 6691.091679637143
Total e: 6197.443774721293
Total e: 5910.549085966008
Total e: 5678.391317973746
Total e: 5478.447318454014
Total e: 5299.388231258417
calculating RMSE ...
FM  dev RMSLE: 0.24197215576549963
calculating RMSE ...
FM_FTRL dev RMSLE: 0.2408992078078851

calculating RMSE ...
 FTRL dev RMSLE: 0.24041738076612498
Total e: 6694.977462749721
Total e: 6201.135074918874
Total e: 5913.840993519399
Total e: 5675.960904656893
Total e: 5468.428715351852
Total e: 5279.872363537804
calculating RMSE ...
FM  dev RMSLE: 0.24162872695531198
calculating RMSE ...
FM_FTRL dev RMSLE: 0.2406825046482856
2018-06-23 13:44:26

全数据
calculating RMSE ...
FM  dev RMSLE: 0.23003102622238622
calculating RMSE ...
FM_FTRL dev RMSLE: 0.22870936935371064
2018-06-23 13:11:24



    '''