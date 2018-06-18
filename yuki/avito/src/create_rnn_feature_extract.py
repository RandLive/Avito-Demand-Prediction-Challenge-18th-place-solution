import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Concatenate, SpatialDropout1D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
from keras.engine import InputSpec, Layer
import warnings
warnings.filterwarnings('ignore')
import argparse
from tqdm import tqdm
import gc
import fastText
import sys, os, re, csv, codecs, time
start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import logging
import dill
from utils import *
from nltk.corpus import stopwords
russian_stop = set(stopwords.words('russian'))
stop_2 = set([w for w in open("../tmp/russian_stopwords.txt", "r").readlines()])
russian_stop = russian_stop.union(stop_2)
import string
punctuations = string.punctuation
import traceback


is_trainable = 0
EMBEDDING_FILE = "../model/fasttext_model_without_stop.bin"
EMBEDDING_SIZE = 100
activation = "sigmoid"

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true-y_pred)))

try:
    print("reading data...")
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    submission = pd.read_csv('../input/sample_submission.csv')
    train["text_all"] = train.description.fillna("") + " " + train.title.fillna("")
    test["text_all"] = test.description.fillna("") + " " + test.title.fillna("")

    def inputpreprocess(text):
        t = ' '.join([t for t in text.split() if t not in russian_stop and t not in punctuations and '\n' != t and " " != t])
        t = [s for s in t if "\n" != s and '"' != s]
        text = "".join(t)
        text = re.sub(r"^\s+","",text)
        return text

    train["description"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in train.description.fillna("").values])
    test["description"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in test.description.fillna("").values])
    train["title"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in train.title.fillna("").values])
    test["title"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in test.title.fillna("").values])

    X = (train["description"].fillna("") + " " + train["title"].fillna("")).values
    X_test = (test["description"].fillna("") + " " + test["title"].fillna("")).values
    y = pd.read_csv("../input/train.csv")["deal_probability"].values

    embed_size = EMBEDDING_SIZE
    maxlen_desc = 120
    maxlen_title = 21
    embedding_vectors = {}
    word_idx = {}
    embedding_vectors["UNKNOWN"] = np.zeros(embed_size)
    word_idx["UNKNOWN"] = len(word_idx)

    model = fastText.load_model(EMBEDDING_FILE)#.replace(".vec", ".bin"))
    print("getting embedding vectors...")
    for text in tqdm(list(X) + list(X_test)): # max_length, padding,
        for w in text.split():
            try:
                embedding_vectors[w] = model.get_word_vector(w)
                if w in word_idx:
                    continue
                else:
                    word_idx[w] = len(word_idx)
            except:
                pass
    idx_to_word = {v:k for k,v in word_idx.items()}
    del model

    train_test_index_list_1 = []
    train_test_index_list_2 = []
    test_index_list = []
    train_size = X.shape[0]
    print("getting sentence idx list...")
    for text in tqdm(train["description"].fillna("").tolist()+test["description"].fillna("").tolist()):
        sentence_list = []
        for w in text.split():
            try:
                sentence_list.append(word_idx[w])
            except:
                sentence_list.append(word_idx["UNKNOWN"])


        if len(sentence_list) >= maxlen_desc:
            sentence_list = sentence_list[:maxlen_desc]
        else:
            sentence_list += [len(word_idx)-1] * (maxlen_desc-len(sentence_list))
        train_test_index_list_1.append(sentence_list)

    for text in tqdm(train["title"].fillna("").tolist()+test["title"].fillna("").tolist()):
        sentence_list = []
        for w in text.split():
            try:
                sentence_list.append(word_idx[w])
            except:
                sentence_list.append(word_idx["UNKNOWN"])


        if len(sentence_list) >= maxlen_title:
            sentence_list = sentence_list[:maxlen_title]
        else:
            sentence_list += [len(word_idx)-1] * (maxlen_title-len(sentence_list))
        train_test_index_list_2.append(sentence_list)


    X_1 = np.array(train_test_index_list_1[:train_size])
    X_test_1 = np.array(train_test_index_list_1[train_size:])
    X_2 = np.array(train_test_index_list_2[:train_size])
    X_test_2 = np.array(train_test_index_list_2[train_size:])
    max_features = len(word_idx)
    embedding_matrix = np.zeros((max_features, embed_size))
    print("preparing embedding matrix...")
    for i in tqdm(list(range(max_features))):
        embedding_matrix[i,:] = embedding_vectors[idx_to_word[i]]
    del train_test_index_list_1, train_test_index_list_2, word_idx, idx_to_word, embedding_vectors, sentence_list, X, X_test

    from sklearn.model_selection import StratifiedKFold, KFold
    nsplits = 5
    with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)
    val_score = []
    result = np.zeros((X_test_1.shape[0], 1))
    cnt = 0
    oof_valid = np.zeros((X_1.shape[0], 1))
    batch_size = 128
    for ix_train, ix_valid in kfolds:
        print("============ROUND{}==============".format(cnt+1))
        cnt+=1
        X_train_1, X_val_1 = X_1[ix_train,:], X_1[ix_valid, :]
        X_train_2, X_val_2 = X_2[ix_train,:], X_2[ix_valid, :]

        y_train = y[ix_train]
        y_val = y[ix_valid]
        X_train = {"description": X_train_1, "title": X_train_2}
        X_val = {"description": X_val_1, "title": X_val_2}

        max_num_epoch = 6
        early_stopping =EarlyStopping(monitor='val_loss', patience=2)
        bst_model_path = 'rnn.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

        inp1 = Input(shape=(maxlen_desc, ), name="description")
        inp2 = Input(shape=(maxlen_title, ), name="title")
        emb1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=is_trainable)(inp1)
        emb2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=is_trainable)(inp2)
        x = concatenate([emb1, emb2], axis=1)
        x = SpatialDropout1D(0.2)(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        outp = Dense(1, activation=activation)(conc)

        model = Model(inputs=[inp1, inp2], outputs=outp)
        print("======================================================")
        print("start adam optimizer...")
        print("======================================================")
        model.compile(loss=root_mean_squared_error,
                      optimizer="adam",
                      metrics=[root_mean_squared_error])
        model.fit(X_train, y_train, batch_size=batch_size, epochs=max_num_epoch, verbose=1
                    , shuffle=True, validation_data=(X_val, y_val)
                    , callbacks=[early_stopping, model_checkpoint])
        model.load_weights(bst_model_path)

        # # Finetuning
        # inp1 = Input(shape=(maxlen_desc, ), name="description")
        # inp2 = Input(shape=(maxlen_title, ), name="title")
        # emb1 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=1)(inp1)
        # emb2 = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=1)(inp2)
        # x = concatenate([emb1, emb2], axis=1)
        # x = SpatialDropout1D(0.15)(x)
        # x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
        # avg_pool = GlobalAveragePooling1D()(x)
        # max_pool = GlobalMaxPooling1D()(x)
        # conc = concatenate([avg_pool, max_pool])
        # outp = Dense(1, activation=activation)(conc)
        #
        # model = Model(inputs=[inp1, inp2], outputs=outp)
        # print("======================================================")
        # print("Fine Tuning...")
        # print("======================================================")
        # model.compile(loss=root_mean_squared_error,
        #               optimizer="adam",
        #               metrics=[root_mean_squared_error])
        # model.load_weights(bst_model_path)
        # model.fit(X_train, y_train, batch_size=batch_size, epochs=max_num_epoch, verbose=1
        #             , shuffle=True, validation_data=(X_val, y_val)
        #             , callbacks=[early_stopping, model_checkpoint])
        # model.load_weights(bst_model_path)


        X_test = {"description": X_test_1, "title": X_test_2}
        result += model.predict(X_test, batch_size=64) / nsplits
        # valication_score = val_func(y_val, model.predict(X_val))
        oof_valid[ix_valid, :] = model.predict(X_val)
        # print("======================================================")
        # print("======================================================")
        # print("===========val score : ", valication_score)
        # print("=======================done===========================")
        # print("=======================done===========================")
        # val_score.append(valication_score)
        del inp1, inp2, emb1, emb2, x, avg_pool, max_pool, conc, outp, model, model_checkpoint; gc.collect()
        del X_train, X_val, y_train, y_val; gc.collect()
        K.clear_session()
        gc.collect()

    df_out = pd.DataFrame(result, columns=["oof_rnn_{}_".format(activation)])
    to_parquet(df_out, "../features/oof_rnn_{}_test.parquet".format(activation))

    df_out = pd.DataFrame(oof_valid, columns=["oof_rnn_{}_".format(activation)])
    to_parquet(df_out, "../features/oof_rnn_{}_train.parquet".format(activation))
except Exception as e:
    notify_line(traceback.format_exc())
