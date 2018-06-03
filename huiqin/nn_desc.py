#encoding=utf-8
'''
用来测试新的预处理函数
新的预处理函数
RMSE: 0.2513543867752114
RMSE: 0.2509765066803424
RMSE: 0.251205915150865
RMSE: 0.2511475928439091
RMSE: 0.25244284405317813
平均:0.251354387

--------------------------
旧的预处理函数
RMSE: 0.2521512393579412
RMSE: 0.25383218397238644
RMSE: 0.2513514009178597
RMSE: 0.24926505621887932
RMSE: 0.2515168616456848
平均:0.251623348

'''
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pandas as pd
from keras.preprocessing import text, sequence
import numpy as np
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EMBEDDING_FILE = 'C:/datas/wiki.ru.vec'
TRAIN_CSV = '../input/train.csv'
TEST_CSV = '../input/test.csv'

max_features = 100000
maxlen = 100
embed_size = 300

train = pd.read_csv(TRAIN_CSV, index_col = 0,nrows=10000)
labels = train[['deal_probability']].copy()
train = train[['description']].copy()

tokenizer = text.Tokenizer(num_words=max_features)
print('fitting tokenizer')
def preprocess1(string):
    '''
    旧的预处理函数
    :param string:
    :return:
    '''
    #去掉一些特殊符号
    string=str(string)
    string = re.sub(r'\"', ' ', string)
    string = re.sub(r'\n', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    # string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)
    string = re.sub(r'\,', ' ', string)
    string = re.sub(r'\.', ' ', string)
    return string
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

    text = re.sub(r'ip4200', ' canon4200 ', text)
    text = re.sub(r'ip4300', ' canon4300 ', text)
    text = re.sub(r'ip4500', ' canon4500 ', text)
    text = re.sub(r'mp500', ' canon500 ', text)
    text = re.sub(r'mp530', ' canon530 ', text)
    text = re.sub(r'mp610', ' canon610 ', text)
    #以上没有影响
    #

    text = re.sub(r'hamburg', ' hamburg ', text)
    text = re.sub(r'lumia', ' lumia ', text)
    text = re.sub(r'seagate', ' seagate ', text)

    text = re.sub(r'512mb', ' 512mb ', text)
    text = re.sub(r'128mb', ' 128mb ', text)
    text = re.sub(r'256mb', ' 256mb ', text)
    text = re.sub(r'16gb', ' 16gb ', text)
    text = re.sub(r'32gb', ' 32gb ', text)
    text = re.sub(r'64gb', ' 64gb ', text)
    text = re.sub(r'500gb', ' 500gb ', text)
    text = re.sub(r'260gb', ' 260gb ', text)
    text = re.sub(r'250gb', ' 250gb ', text)
    text = re.sub(r'320gb', ' 320gb ', text)
    text = re.sub(r'1000gb', ' 1000gb ', text)
    text = re.sub(r'20gb', ' 20gb ', text)


    #
    text = re.sub(r'\®', ' ', text)
    text = re.sub(r'intel', ' intel ', text)

    text = re.sub(r'canon', ' canon ', text)
    text = re.sub(r'adidas', ' adidas ', text)
    text = re.sub(r'gucci', ' gucci ', text)
    #没有什么改进，不变
    text = re.sub(r'\\u200b', '  ', text)
    text = re.sub(r'\\u200d', '  ', text)


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
    text = re.sub(r'panasoni', ' panasonic ', text)
    #mean rmse is: 0.2369177999350502
    text = re.sub(r'compac', ' compac ', text)

    # text = re.sub(r'платье', ' платье ', text)
    # text = re.sub(r'продам', ' продам ', text)
    text = re.sub(r'tomy', ' tomy ', text)
    text = re.sub(r'✔', ' ', text)
    text = re.sub(r'👌', ' ', text)
    text = re.sub(r'💰', ' ', text)
    text = re.sub(r'❤', ' ', text)
    text = re.sub(r'htc', ' htc ', text)


    text = re.sub(r'gtx', ' gtx ', text)
    text = re.sub(r"(\\u[0-9A-Fa-f]+)",r"", text)
    text = re.sub(r"===",r" ", text)
    # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = " ".join(text.split())
    return text

train['description'] = train['description'].astype(str)
train['description'] = train['description'].apply(text_preprocessing_v2)
tokenizer.fit_on_texts(list(train['description'].fillna('NA').values))


print('getting embeddings')
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
# embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMBEDDING_FILE,encoding='utf-8')))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
print("nb_words",nb_words)

def make_glovevec(glovepath, max_features, embed_size, word_index, veclen=300):
    embeddings_index = {}
    f = open(glovepath, encoding='utf-8')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# embedding_matrix = np.zeros((nb_words, embed_size))
# for word, i in tqdm(word_index.items()):
#     if i >= max_features: continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector
#
# del embeddings_index

# embedding_matrix = make_glovevec(EMBEDDING_FILE,max_features,embed_size,word_index)
X_train, X_valid, y_train, y_valid = train_test_split(train['description'].values, labels['deal_probability'].values, test_size = 0.1, random_state = 23)
del train
print('convert to sequences')
X_train = tokenizer.texts_to_sequences(X_train)
X_valid = tokenizer.texts_to_sequences(X_valid)

print('padding')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model():
    inp = Input(shape = (maxlen, ))
    # emb = Embedding(nb_words, embed_size, weights = [embedding_matrix],
    #                 input_length = maxlen, trainable = False)(inp)
    emb = Embedding(nb_words, embed_size,
                    input_length=maxlen)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(CuDNNGRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation = "sigmoid")(main)

    model = Model(inputs = inp, outputs = out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    model.summary()
    return model

EPOCHS = 10

model = build_model()
file_path = "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
history = model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 2, callbacks = [check_point])
model.load_weights(file_path)
prediction = model.predict(X_valid)
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, prediction)))


# test = pd.read_csv(TEST_CSV, index_col = 0)
# test = test[['description']].copy()
#
# test['description'] = test['description'].astype(str)
# X_test = test['description'].values
# X_test = tokenizer.texts_to_sequences(X_test)
#
# print('padding')
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# prediction = model.predict(X_test,batch_size = 128, verbose = 2)

# sample_submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv', index_col = 0)
# submission = sample_submission.copy()
# submission['deal_probability'] = prediction
# submission.to_csv('submission.csv')
'''
新的预处理函数
RMSE: 0.2513543867752114
RMSE: 0.2509765066803424
RMSE: 0.251205915150865
RMSE: 0.2511475928439091
RMSE: 0.25244284405317813
平均:0.251354387

--------------------------
旧的预处理函数
RMSE: 0.2521512393579412
RMSE: 0.25383218397238644
RMSE: 0.2513514009178597
RMSE: 0.24926505621887932
RMSE: 0.2515168616456848
平均:0.251623348

--------------------------
RMSE: 0.2519665484913265
RMSE: 0.24847508465534324
RMSE: 0.25253833079928767
'''