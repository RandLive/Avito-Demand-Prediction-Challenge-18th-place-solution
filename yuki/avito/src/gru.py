import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, Concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, CuDNNGRU
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
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


parser = argparse.ArgumentParser()
parser.add_argument('--is_trainable', '-it', default=0, help='embedding matrix is trainable')
parser.add_argument('--embedding_file', '-ef', default='', help='embedding file')
parser.add_argument('--embedding_size', '-es', default=100, help='embedding size')
parser.add_argument('--model_type', '-mt', default="gru", help='model type')
parser.add_argument('--input', '-i', default="preprocessed", help='input')
parser.add_argument('--output', '-o', default="gru_submit", help='output')
parser.add_argument('--activation', '-a', default="linear", help='activation')
args = parser.parse_args()
is_trainable = int(args.is_trainable)
EMBEDDING_FILE = args.embedding_file
EMBEDDING_SIZE = int(args.embedding_size)
inputfile = args.input
outputfile = args.output
modeltype = args.model_type
activation = args.activation

def val_func(y_val, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_val, y_pred))

try:
    print("reading data...")
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    submission = pd.read_csv('../input/sample_submission.csv')
    train["text_all"] = train.description.fillna("") + " " + train.param_1.fillna("") \
                + " " + train.param_2.fillna("") + " " + train.param_3.fillna("") \
                + " " + train.title.fillna("")
    test["text_all"] = test.description.fillna("") + " " + test.param_1.fillna("") \
                + " " + test.param_2.fillna("") + " " + test.param_3.fillna("") \
                + " " + test.title.fillna("")

    if inputfile == "preprocessed":
        def inputpreprocess(text):
            t = ' '.join([t for t in text if t not in russian_stop and t not in punctuations and '\n' != t and " " != t])
            t = [s for s in t if "\n" != s and '"' != s]
            text = "".join(t)
            text = re.sub(r"^\s+","",text)
            return text
    elif input == "raw":
        def inputpreprocess(text):
            return text

    train["text_all"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in train.text_all.fillna("fillna").values])
    test["text_all"] = Parallel(n_jobs=-1)([delayed(inputpreprocess)(v) for v in test.text_all.fillna("fillna").values])

    train_features, test_features, y = read_train_test_data_nonlp()
    scaler = MinMaxScaler()
    scaler.fit(np.vstack([train_features, test_features]))
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    X = train["text_all"].fillna("fillna").values
    X_test = test["text_all"].fillna("fillna").values


    if EMBEDDING_SIZE==300:
        """
        X
        X_test
        max_features
        max_len
        embed_size
        embedding_matrix
        """
        max_features = 130000
        maxlen = 200
        embed_size = EMBEDDING_SIZE
        print("tokenizing text...")
        tokenizer = Tokenizer(num_words=max_features
                                    ,filters='"#$%&()+,-./:;<=>@[\\]^_`{|}~\t\n'
                                    ,lower=True
                                    ,split=" "
                                    ,char_level=False)
        tokenizer.fit_on_texts(list(X) + list(X_test))
        X = tokenizer.texts_to_sequences(X)
        X_test = tokenizer.texts_to_sequences(X_test)
        X = sequence.pad_sequences(X, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)


        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

        word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words, embed_size))
        print("preparing embedding matrix...")
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    else:
        embed_size = EMBEDDING_SIZE
        maxlen = 300
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

        train_test_index_list = []
        train_size = X.shape[0]
        print("getting sentence idx list...")
        for text in tqdm(list(X) + list(X_test)):
            sentence_list = []
            for w in text.split():
                try:
                    sentence_list.append(word_idx[w])
                except:
                    sentence_list.append(word_idx["UNKNOWN"])


            if len(sentence_list) >= maxlen:
                sentence_list = sentence_list[:maxlen]
            else:
                sentence_list += [len(word_idx)-1] * (maxlen-len(sentence_list))
            train_test_index_list.append(sentence_list)

        X = np.array(train_test_index_list[:train_size])
        X_test = np.array(train_test_index_list[train_size:])
        max_features = len(word_idx)
        embedding_matrix = np.zeros((max_features, embed_size))
        print("preparing embedding matrix...")
        for i in tqdm(list(range(max_features))):
            embedding_matrix[i,:] = embedding_vectors[idx_to_word[i]]
        del train_test_index_list, word_idx, idx_to_word, embedding_vectors, sentence_list


    from sklearn.model_selection import StratifiedKFold, KFold
    nsplits = 5
    with open("../tmp/oof_index.dat", "rb") as f:
        kfolds = dill.load(f)
    val_score = []
    result = np.zeros((X_test.shape[0], 1))
    cnt = 0
    oof_valid = np.zeros((X.shape[0], 1))
    batch_size = 1024
    for ix_train, ix_valid in kfolds:
        print("============ROUND{}==============".format(cnt+1))
        cnt+=1
        X_train = X[ix_train,:]
        X_val = X[ix_valid, :]
        y_train = y[ix_train]
        y_val = y[ix_valid]
        hand_features_train = train_features[ix_train,:]
        hand_features_val = train_features[ix_valid,:]
        X_train = {"text": X_train, "crafted_features": hand_features_train}
        X_val = {"text": X_val, "crafted_features": hand_features_val}

        if modeltype=="gru":
            max_num_epoch = 5
            inp = Input(shape=(maxlen, ), name="text")
            crafted_features = Input(shape=[train_features.shape[1]], name="crafted_features")
            # x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
            x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=is_trainable)(inp)
            x = SpatialDropout1D(0.25)(x)
            x = Bidirectional(GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            conc = concatenate([avg_pool, max_pool, crafted_features])
            z = Dense(64)(conc)
            z = Dropout(0.1)(z)
            outp = Dense(1, activation=activation)(z)
            # outp = Dense(1, activation="linear")(z)

            model = Model(inputs=[inp, crafted_features], outputs=outp)
            print("======================================================")
            print("start adam optimizer...")
            print("======================================================")
            model.compile(loss='mean_squared_error',
                          optimizer="adam",
                          metrics=['accuracy'])
            print(model.summary())
            early_stopping =EarlyStopping(monitor='val_loss', patience=4)
            bst_model_path = 'gru_{}.h5'.format(outputfile)

            # RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        elif modeltype=="bigrucnn":
            bst_model_path = "best_model{}.hdf5".format(outputfile)
            model_checkpoint = ModelCheckpoint(bst_model_path, monitor = "val_loss", verbose = 1,
                                          save_best_only = True, mode = "min")

            early_stopping = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)
            max_num_epoch = 10
            inp = Input(shape = (maxlen,), name="text")
            crafted_features = Input(shape=[train_features.shape[1]], name="crafted_features")
            x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = is_trainable)(inp)
            x = SpatialDropout1D(0.2)(x)

            x = Bidirectional(GRU(256, return_sequences = True, dropout=0.1, recurrent_dropout=0.1))(x)
            x = Conv1D(128, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            x = concatenate([avg_pool, max_pool,crafted_features])
            x = Dense(128)(x)
            x = Dropout(0.1)(z)
            x = Dense(1, activation = activation)(x)
            model = Model(inputs = [inp, crafted_features], outputs = x)
            model.compile(loss = "mean_squared_error"
                        , optimizer = "adam"#Adam(lr = 1e-3, decay = 0)
                        , metrics = ["accuracy"])
        elif modeltype=="cnnbase":
            batch_size = 64
            max_num_epoch = 100
            X_train = X[ix_train,:]
            X_val = X[ix_valid, :]
            y_train = y[ix_train]
            y_val = y[ix_valid]
            from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
            from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
            filter_sizes = [1,2,3,5]
            num_filters = 16
            bst_model_path = "cnn_{}.hdf5".format(outputfile)
            model_checkpoint = ModelCheckpoint(bst_model_path, monitor = "val_loss", verbose = 1,
                                          save_best_only = True, mode = "min")
            early_stopping = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)
            inp = Input(shape=(maxlen, ))
            x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
            x = SpatialDropout1D(0.5)(x)
            x = Reshape((maxlen, embed_size, 1))(x)

            conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                            activation='elu')(x)
            conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                            activation='elu')(x)
            conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                            activation='elu')(x)
            conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                            activation='elu')(x)

            maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
            maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
            maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
            maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

            z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
            z = Flatten()(z)
            z = Dropout(0.3)(z)
            z = Dense(256)(z)
            z = Dropout(0.3)(z)

            outp = Dense(1, activation=activation)(z)

            model = Model(inputs=inp, outputs=outp)
            model.compile(loss='mean_squared_error',
                          optimizer='adam',#Adam(lr = 0.0003, decay = 0),
                          metrics=['accuracy'])
        elif modeltype=="dpcnn":
            X_train = X[ix_train,:]
            X_val = X[ix_valid, :]
            y_train = y[ix_train]
            y_val = y[ix_valid]
            from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
            from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
            filter_nr = 32
            filter_size = 3
            max_pool_size = 3
            max_pool_strides = 2
            dense_nr = 128
            spatial_dropout = 0.3
            dense_dropout = 0.5
            train_embed = False
            max_num_epoch = 100
            batch_size = 64
            bst_model_path = "dpcnn_{}.hdf5".format(outputfile)
            model_checkpoint = ModelCheckpoint(bst_model_path, monitor = "val_loss", verbose = 1,
                                          save_best_only = True, mode = "min")
            early_stopping = EarlyStopping(monitor = "val_loss", mode = "min", patience = 6)

            comment = Input(shape=(maxlen,))
            emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
            emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)
            block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
            block1 = BatchNormalization()(block1)
            block1 = PReLU()(block1)

            #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
            #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
            resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment)
            resize_emb = PReLU()(resize_emb)

            block1_output = add([block1, resize_emb])
            block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
            block2 = BatchNormalization()(block2)
            block2 = PReLU()(block2)
            block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
            block2 = BatchNormalization()(block2)
            block2 = PReLU()(block2)

            block2_output = add([block2, block1_output])
            block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

            block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
            block3 = BatchNormalization()(block3)
            block3 = PReLU()(block3)
            block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
            block3 = BatchNormalization()(block3)
            block3 = PReLU()(block3)

            block3_output = add([block3, block2_output])
            block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

            block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
            block4 = BatchNormalization()(block4)
            block4 = PReLU()(block4)
            block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
            block4 = BatchNormalization()(block4)
            block4 = PReLU()(block4)

            output = add([block4, block3_output])
            output = GlobalMaxPooling1D()(output)
            output = Dense(dense_nr, activation='linear')(output)
            output = BatchNormalization()(output)
            output = PReLU()(output)
            output = Dropout(dense_dropout)(output)
            output = Dense(1, activation=activation)(output)

            model = Model(comment, output)


            model.compile(loss='mean_squared_error',
                        optimizer='adam',#Adam(lr = 0.0003, decay = 0),
                        metrics=['accuracy'])

        elif modeltype=="lstm":
            X_train = X[ix_train,:]
            X_val = X[ix_valid, :]
            y_train = y[ix_train]
            y_val = y[ix_valid]
            batch_size = 32
            max_num_epoch = 5
            early_stopping =EarlyStopping(monitor='val_loss', patience=2)
            bst_model_path = 'lstm_{}.h5'.format(outputfile)
            # RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
            inp = Input(shape=(maxlen,))

            x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
            x = SpatialDropout1D(0.25)(x)

            x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
            x = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)

            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)
            x = concatenate([avg_pool, max_pool])

            out = Dense(1, activation=activation)(x)

            model = Model(inp, out)
            model.compile(loss='mean_squared_error'
                        , optimizer='adam'
                        , metrics=['accuracy'])
        elif modeltype=="rnncnn":
            X_train = X[ix_train,:]
            X_val = X[ix_valid, :]
            y_train = y[ix_train]
            y_val = y[ix_valid]
            batch_size = 64
            max_num_epoch = 8
            early_stopping =EarlyStopping(monitor='val_loss', patience=2)
            bst_model_path = 'rnncnn_{}.h5'.format(outputfile)
            # RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
            model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
            inp = Input(shape=(maxlen,))
            em = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
            x = SpatialDropout1D(0.2)(em)
            x = Bidirectional(LSTM(40, return_sequences = True, recurrent_dropout=0.1))(x)
            x = Conv1D(60, kernel_size = 3, padding = "valid", activation="relu", strides=1)(x)
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPooling1D()(x)

            x2 = SpatialDropout1D(0.2)(em)
            x2 = Bidirectional(LSTM(80, return_sequences = True, recurrent_dropout=0.1))(x2)
            x2 = Conv1D(120, kernel_size = 2, padding = "valid", activation="relu", strides=1)(x2)
            avg_pool2 = GlobalAveragePooling1D()(x2)
            max_pool2 = GlobalMaxPooling1D()(x2)

            conc = concatenate([avg_pool, max_pool, avg_pool2, max_pool2])
            outp= Dense(1, activation = activation)(conc)
            model = Model(inputs = inp, outputs = outp)
            model.compile(loss = "mean_squared_error"
                        , optimizer = "adam"#Adam(lr = 1e-3, decay = 0)
                        , metrics = ["accuracy"])



        model.fit(X_train, y_train, batch_size=batch_size, epochs=max_num_epoch, verbose=1
                    , shuffle=True, validation_data=(X_val, y_val)
                    , callbacks=[early_stopping, model_checkpoint])
        model.load_weights(bst_model_path)


        if modeltype=="dpcnn" or modeltype=="cnnbase" or modeltype=="lstm" or modeltype=="rnncnn":
            result += model.predict(X_test, batch_size=64) / nsplits
        else:
            result += model.predict({"text": X_test, "crafted_features":test_features}, batch_size=64) / nsplits
        valication_score = val_func(y_val, model.predict(X_val))
        oof_valid[ix_valid, :] = model.predict(X_val)
        print("======================================================")
        print("======================================================")
        print("===========val score : ", valication_score)
        print("=======================done===========================")
        print("=======================done===========================")
        val_score.append(valication_score)


        del X_train, X_val, y_train, y_val; gc.collect()

    df_out = pd.DataFrame(result, columns=target_labels)
    df_out["id"] = test["id"].values
    to_parquet(df_out, "../features/{}_{}_{}_{}_test.csv".format(outputfile, input, activation, np.mean(val_score)))

    df_out = pd.DataFrame(oof_valid, columns=target_labels)
    df_out["id"] = pd.read_csv('../input/train_preprocessed.csv')["id"].values
    to_parquet(df_out, "../features/{}_{}_{}_{}_train.csv".format(outputfile, input, activation, np.mean(val_score)))
except Exception as e:
    notify_line(traceback.format_exc())
