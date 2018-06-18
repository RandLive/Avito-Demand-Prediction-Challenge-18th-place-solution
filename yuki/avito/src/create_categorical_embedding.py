import numpy as np
import pandas as pd
import os
import gc
from utils import *
from sklearn.decomposition import NMF
import gensim
from gensim.models.doc2vec import LabeledSentence
from gensim import corpora, models
from collections import defaultdict
from tqdm import tqdm

# Embedding
train = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["deal_probability", "image", "image_top_1"],axis=1)
test = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["image", "image_top_1"],axis=1)
train_active = pd.read_csv('../input/train_active.csv', index_col = "item_id", parse_dates = ["activation_date"])
test_active = pd.read_csv('../input/test_active.csv', index_col = "item_id", parse_dates = ["activation_date"])
n_train = train.shape[0]
n_test = test.shape[0]
n_train_acitive = train_active.shape[0]
n_test_active = train.shape[0]
df = pd.concat([train, test, train_active, test_active])
target_user_ids = set(train.user_id).union(set(test.user_id))
del train, test, train_active, test_active; gc.collect()
df["dayofweek"] = df.activation_date.dt.weekday

def lda_5(df, idx, col1, col2):
    tmp = df[[col1, col2]].fillna("NAN").astype(str)
    dic = defaultdict(list)
    for c1, c2 in tmp.values:
        dic[c1].append(c2)
    text = []
    idx = []
    for k,v in dic.items():
        idx.append(k)
        text.append(v)
    df_out = pd.DataFrame()
    df_out[col1] = idx
    num_topics = 3
    dictionary = corpora.Dictionary(text)
    corpus = [dictionary.doc2bow(t) for t in text]
    print("start lda...")
    lda = models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=3000)
    vecs = np.zeros((len(corpus), lda.num_topics))
    for i, topics_per_document in enumerate(lda[corpus]):
        for topic_num, prob in topics_per_document:
            vecs[i, topic_num] = prob

    df_lda = pd.DataFrame(vecs, index=idx, columns=["categorical_embedding_lda_{}_{}_{}".format(col1, col2, i+1) for i in range(num_topics)])
    df_lda.index.name = col1
    df_lda = df_lda.reset_index()
    if col1=="user_id":
        df_lda = df_lda[df_lda.user_id.isin(target_user_ids)]

    print("LDA_{}_{}:".format(col1, col2))
    print(df_lda.head())
    return df_lda

def nmf_5(mat, idx, col1, col2):
    n_components = 3
    nmf = NMF(n_components=n_components)
    mat = nmf.fit_transform(mat)
    df_nmf = pd.DataFrame(mat, index=idx, columns=["categorical_embedding_nmf_{}_{}_{}".format(col1, col2, i+1) for i in range(n_components)])

    df_nmf.index.name = col1
    df_nmf = df_nmf.reset_index()
    if col1=="user_id":
        df_nmf = df_nmf[df_nmf.user_id.isin(target_user_ids)]

    print("NMF_{}_{}:".format(col1, col2))
    print(df_nmf.head())
    return df_nmf


def save_categorical_embedding(col1, col2):
    """
    return categorical embedding(LDA, NMF) by col1 and col2
    """

    df_piv = pd.pivot_table(df[[col1, col2]].astype(str).fillna("NAN"), index=col1, columns=col2, aggfunc=len, fill_value=0)
    idx = df_piv.index
    lda_mat = lda_5(df, idx, col1, col2)
    to_parquet(lda_mat, "../features/fe_categorical_embedding_lda_col1_{}_col2_{}.parquet".format(col1, col2))
    nmf_mat = nmf_5(df_piv, idx, col1, col2)
    to_parquet(nmf_mat, "../features/fe_categorical_embedding_nmf_col1_{}_col2_{}.parquet".format(col1, col2))
    # df_out = pd.concat([nmf_mat, lda_mat],axis=1)
    # df_out = df_out.reset_index()


target_cols = [
    # ("city", "category_name"),
    # ("user_id", "category_name"),
    # # ("user_id", "parent_category_name"),
    # ("user_id", "param_1"),
    # ("user_id", "param_2"),
    # ("category_name", "user_id"),
    # ("category_name", "city")
    # ("region", "category_name"),
    # ("region", "param_1"),
    # ("region", "user_id"),
    # ("param_1", "region"),
    # ("param_1", "user_id"),
    # ("param_2", "region"),
    # ("param_2", "user_id"),
    # ("param_3", "region"),
    # ("param_3", "user_id"),
    # # ("parent_category_name", "param_1"),
    # # ("parent_category_name", "param_2"),
    # # ("user_type", "parent_category_name"),
    # # ("user_type", "param_1"),
    # # ("user_type", "param_2"),
    #
    # ("city", "image_top_1"),
    # ("user_id", "image_top_1"),
    # ("region", "image_top_1"),
    # # ("user_type", "image_top_1"),
    # # ("parent_category_name", "image_top_1"),
    # ("param_1", "image_top_1"),
    # ("param_2", "image_top_1"),
    # ("param_3", "image_top_1"),
    # ("image_top_1", "user_id"),
    # ("image_top_1", "city"),
    # ("image_top_1", "param_1"),
    # ("image_top_1", "param_2"),
    # ("image_top_1", "param_3"),
    # # ("image_top_1", "parent_category_name")
    ("parent_category_name", "user_type"),
    ("param_1", "user_type"),
    ("param_2", "user_type"),
    ("image_top_1", "user_type")
]
for col1, col2 in tqdm(target_cols):
    if col1=="image_top_1" or col2=="image_top_1":
        train = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["deal_probability", "image"],axis=1)
        test = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"]).drop(["image"],axis=1)
        df = pd.concat([train, test])
        del train, test; gc.collect()
    print("{}_{}".format(col1, col2))
    save_categorical_embedding(col1, col2)
