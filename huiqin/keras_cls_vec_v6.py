'''
keras 初始版本
使用词嵌入方式处理输入的文本信息
lb 0.2261
'''
import time

notebookstart = time.time()

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc

print("Data:\n", os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import *
import re
from tqdm import tqdm
print("\nData Load Stage")
# training = pd.read_csv('../input/train.csv',nrows=10000, index_col="item_id", parse_dates=["activation_date"])
# testing = pd.read_csv('../input/test.csv',nrows=10000, index_col="item_id", parse_dates=["activation_date"])

training = pd.read_csv('../input/train.csv', index_col="item_id", parse_dates=["activation_date"])
testing = pd.read_csv('../input/test.csv', index_col="item_id", parse_dates=["activation_date"])

# training = pd.read_csv('../input/train_translated.csv', index_col="item_id", parse_dates=["activation_date"])
# testing = pd.read_csv('../input/test_translated.csv', index_col="item_id", parse_dates=["activation_date"])
traindex = len(training)
testdex = testing.index

print("traindex",type(traindex))
y = training.deal_probability.copy()
training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training, testing], axis=0)
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type','item_seq_number','activation_weekday']
# for c in tqdm(agg_cols):
#     gp = training.groupby(c)['deal_probability']
#     mean = gp.mean()
#     std  = gp.std()
#     df[c + '_deal_probability_avg'] = df[c].map(mean)
#     df[c + '_deal_probability_std'] = df[c].map(std)
#
# for c in tqdm(agg_cols):
#     gp = training.groupby(c)['price']
#     mean = gp.mean()
#     df[c + '_price_avg'] = df[c].map(mean)

del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("\nCreate Time Variables")
df["Year"] = df["activation_date"].dt.year
df["Date of Year"] = df['activation_date'].dt.dayofyear  # Day of Year
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day
df["Quarter"] = df['activation_date'].dt.quarter
#新特征
df['image_available'] = df['image'].map(lambda x: 1 if len(str(x))>0 else 0)
df['description_cat'] = df['description'].map(lambda x: 1 if len(str(x))>0 else 0)
df['param_1_cat'] = df['param_1'].map(lambda x: 1 if len(str(x))>0 else 0)
df['param_2_cat'] = df['param_2'].map(lambda x: 1 if len(str(x))>0 else 0)
df['param_3_cat'] = df['param_3'].map(lambda x: 1 if len(str(x))>0 else 0)

# Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type",
               "image_top_1"
               ,'param_1_cat','param_2_cat','param_3_cat','description_cat','image_available'
               ]
messy_categorical = ["param_1", "param_2", "param_3",
                     ]  # Need to find better technique for these

df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']),
    str(row['param_2']),
    str(row['param_3'])]),axis=1) # Group Param Features
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

print("Encoding :", categorical + messy_categorical)

from keras.preprocessing import text, sequence
c='title'
df[c + '_len'] = df[c].map(lambda x: len(str(x))).astype(np.uint8) #Lenth
df[c + '_wc'] = df[c].map(lambda x: len(str(x).split(' '))).astype(np.uint8) #Word Count
#新特征
df[c+'_capitals'] = df[c].apply(lambda comment: sum(1 for c1 in comment if c1.isupper()))
df[c+'_num_symbols'] = df[c].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))

c='description'
df[c].fillna('na',inplace=True)
df[c + '_len'] = df[c].map(lambda x: len(str(x))).astype(np.uint8) #Lenth
df[c + '_wc'] = df[c].map(lambda x: len(str(x).split(' '))).astype(np.uint8) #Word Count
#新特征
df[c+'_capitals'] = df[c].apply(lambda comment: sum(1 for c1 in comment if c1.isupper()))
df[c+'_num_symbols'] = df[c].apply(
        lambda comment: sum(comment.count(w) for w in '*&$%'))




df['desc']=df['title']+df['description']

#文本数据预处理
def preprocess1(string):
    '''

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

df['desc']=df['desc'].apply(preprocess1)

df['text_feat']=df['text_feat'].apply(preprocess1)

df['price'].fillna(0,inplace=True)
num_features = ["title_len", "title_wc", "description_len", "description_wc"
    ,'price', 'item_seq_number'
    ,"title_capitals", "title_num_symbols","description_capitals", "description_num_symbols",
                ]
for c in num_features:
    df[c]=df[c].apply(lambda x:np.log1p(x))
print("log1p for numeric features...")
# cat_features =categorical + messy_categorical
cat_features =categorical
cat_features_hash = [col+"_hash" for col in cat_features]

max_size=15000#0
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    return df

df = feature_hash(df)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_features])
X_cat = np.array(df[cat_features_hash], dtype=np.int)

max_features=200000
embed_size=300
maxlen = 100#200
vec_file="C:/datas/wiki.ru.vec"
# vec_file="C:/datas/wiki.en.vec"
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


tokenizer = text.Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(df['desc']))
tokenizer.fit_on_texts(list(df['desc']+df['text_feat']))
list_tokenized_desc = tokenizer.texts_to_sequences(list(df['desc']))
X_desc = sequence.pad_sequences(list_tokenized_desc, maxlen=maxlen)

list_tokenized_text_feat = tokenizer.texts_to_sequences(list(df['text_feat']))
X_text_feat = sequence.pad_sequences(list_tokenized_text_feat, maxlen=maxlen)

word_index = tokenizer.word_index
print("word_index", len(word_index)) #1512125

start_time=time.time()

print("start to load vec file...")
embedding_vector = make_glovevec(vec_file,
                                 max_features, embed_size, word_index)
print("loading glove vec costs {}".format(time.time()-start_time))

#划分train和test
ex_col = ['item_id','user_id','deal_probability','description','title','mean_y']
col = [c for c in df.columns if c not in ex_col]

print("\n Modeling Stage")
X_train_num = X_num[:traindex]
X_train_cat=X_cat[:traindex]
X_train_words=X_desc[:traindex]
X_train_words2=X_text_feat[:traindex]
# print("x columns",X_num.columns)
print("Training Set shape", X_train_num.shape)

X_test_num = X_num[traindex :]
X_test_cat=X_cat[traindex :]
X_test_words=X_desc[traindex :]
X_test_words2=X_text_feat[traindex :]


del df,X_desc,X_cat,X_num
gc.collect()

# Training and Validation Set
X_train_cat_tr,X_train_cat_val, X_train_num_tr,X_train_num_val,X_train_words_tr,X_train_words_val,X_train_words2_tr,X_train_words2_val, y_train, y_valid = \
    train_test_split(X_train_cat,X_train_num,X_train_words,X_train_words2,y,test_size=0.10,shuffle=False,random_state=1234)

x_train=[X_train_cat_tr, X_train_num_tr, X_train_words_tr,X_train_words2_tr]
x_valid=[X_train_cat_val, X_train_num_val, X_train_words_val,X_train_words2_val]

x_test=[X_test_cat, X_test_num, X_test_words,X_test_words2]
# print("Submission Set Shape: {} Rows, {} Columns".format(*x_test.shape))

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Flatten, concatenate, GlobalAveragePooling1D
from keras.layers import Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU, SpatialDropout1D,CuDNNLSTM
from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
    GlobalMaxPool1D, SpatialDropout1D, CuDNNGRU, Bidirectional, PReLU,GlobalAvgPool1D,CuDNNLSTM
from keras.models import Model, Layer
from keras.optimizers import Adam,RMSprop
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import K
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

class AttLayer(Layer):
  def __init__(self, use_bias=True, activation ='tanh', **kwargs):
    self.init = initializers.get('normal')
    self.use_bias = use_bias
    self.activation = activation
    super(AttLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    assert len(input_shape)==3
    self.W = self.add_weight(name='kernel',
                             shape=(input_shape[-1],1),
                             initializer='normal',
                             trainable=True)
    if self.use_bias:
      self.bias = self.add_weight(name='bias',
                                  shape=(1,),
                                  initializer='zeros',
                                  trainable=True)
    else:
      self.bias = None
    super(AttLayer, self).build(input_shape)

  def call(self, x, mask=None):
    eij = K.dot(x, self.W)
    if self.use_bias:
      eij = K.bias_add(eij, self.bias)
    if self.activation == 'tanh':
      eij = K.tanh(eij)
    elif self.activation =='relu':
      eij = K.relu(eij)
    else:
      eij = eij
    ai = K.exp(eij)
    weights = ai/K.sum(ai, axis=1, keepdims=True)
    weighted_input = x*weights
    return K.sum(weighted_input, axis=1)

  def compute_output_shape(self, input_shape):
    return (input_shape[0], input_shape[-1])

  def get_config(self):
    config = { 'activation': self.activation }
    base_config = super(AttLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def BidLstm(maxlen, max_features, embedding_matrix=None,embed_size=300):
    '''

    :param maxlen:
    :param max_features:
    :param embed_size:
    :param embedding_matrix:
    :return:
    '''
    input_cat = Input((len(cat_features_hash),))
    input_num = Input((len(num_features),))
    input_words = Input((maxlen,))
    input_words2 = Input((maxlen,))

    x_cat = Embedding(max_size, 10)(input_cat)

    x_cat = SpatialDropout1D(0.3)(x_cat)
    x_cat = Flatten()(x_cat)

    # x_words = Embedding(max_features, embed_size,
    #                     )(input_words)
    x_words = Embedding(max_features, embed_size,
                        weights=[embedding_matrix],
                        trainable=False)(input_words)
    x_words2 = Embedding(max_features, embed_size,
                        )(input_words2)
    x_words_veb=SpatialDropout1D(0.35)(x_words2)
    x_words_veb=Bidirectional(CuDNNGRU(40, return_sequences=True))(x_words_veb)

    gl_veb = GlobalMaxPool1D()(x_words_veb)


    x_words = SpatialDropout1D(0.35)(x_words)
    # x_words = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x_words)
    x_words1 = Bidirectional(CuDNNGRU(40, return_sequences=True))(x_words)  # 50

    attenion = Attention(maxlen)(x_words1)
    gl = GlobalMaxPool1D()(x_words1)
    gl_aver = GlobalAvgPool1D()(x_words1)

    x_cat = Dense(200, )(x_cat)
    x_cat = PReLU()(x_cat)
    x_num = Dense(200, )(input_num)
    x_num = PReLU()(x_num)
    # x_num = Dropout(0.25)(x_num)
    # x_num = Embedding(10, 10)(input_num)
    # x_num=Flatten()(x_num)
    x = concatenate([x_cat, x_num, attenion, gl, gl_aver,gl_veb])
    # x = concatenate([x_cat, x_num,attenion])
    #     x = Dense(200, activation="relu")(x)
    x = Dense(100, )(x)
    x = PReLU()(x)
    x = Dense(50, )(x)
    x = PReLU()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words,input_words2], outputs=predictions)
    # model.compile(optimizer=Adam(0.0001, decay=1e-6),
    #               loss=root_mean_squared_error
    #               ,metrics=['mse']
    #               )
    # model.compile(optimizer="rmsprop", loss=["MSE"], metrics=[root_mean_squared_error])
    model.compile(optimizer="rmsprop", loss=root_mean_squared_error, metrics=['mse'])
    # model.compile(optimizer=RMSprop(lr=0.0005,decay=0.01), loss=root_mean_squared_error, metrics=['mse'])
    #     model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    return model

model = BidLstm(maxlen, max_features,
                embedding_vector,
                embed_size)

# Train Model
print("Train nn...")
file_path='simpleRNN_attention_v6.h5'

from sklearn.model_selection import KFold
from keras.callbacks import *

modelstart = time.time()
i = 0
n_folds=1
loss_total = 0
acc_total = 0
pred_test=0.
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True,
                                     mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               epsilon=1e-4,
                               mode='min')
callbacks_list = [checkpoint, early, lr_reduced]

history = model.fit(x_train, y_train,
                    validation_data=(x_valid,y_valid),
                    verbose=2,callbacks=callbacks_list,
          epochs=50, batch_size=512*4)#512*4

model.load_weights(file_path)

# loss, acc = model.evaluate(x_valid, y_valid, verbose=0)
# loss_total += loss
# acc_total += acc
print ("Avg loss = {}, avg acc = {}".format(loss_total/n_folds, acc_total/n_folds))
preds = model.predict(x_test, batch_size=2000*4)
submission = pd.read_csv( "../input/sample_submission.csv")
submission['deal_probability'] = preds
submission.to_csv("gru_sub_v6.csv", index=False)
# catsub = pd.DataFrame(preds, columns=["deal_probability"], index=testdex)
# catsub['deal_probability'].clip(0.0, 1.0, inplace=True)  # Between 0 and 1
# catsub.to_csv("gru_sub_v3.csv", index=False)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))

'''
Epoch 00003: val_loss improved from 0.22470 to 0.22288, saving model to simpleRNN_attention_v5.h5
 - 168s - loss: 0.2175 - mean_squared_error: 0.0473 - val_loss: 0.2229 - val_mean_squared_error: 0.0497
Epoch 4/50

lb 0.2280
Epoch 00003: val_loss improved from 0.22357 to 0.22338, saving model to simpleRNN_attention_v4.h5
 - 166s - loss: 0.2184 - mean_squared_error: 0.0477 - val_loss: 0.2234 - val_mean_squared_error: 0.0499
Epoch 4/50

Model Runtime: 19.40 Minutes
Notebook Runtime: 24.86 Minutes

使用翻译过的英语和英语词向量
 val_loss: 0.2243 - val_mean_squared_error: 0.0504
lb 0.228
Epoch 00002: val_loss improved from 0.22606 to 0.22450, saving model to simpleRNN_attention_v3.h5
 - 140s - loss: 0.2243 - mean_squared_error: 0.0503 - val_loss: 0.2245 - val_mean_squared_error: 0.0504
test_size=0.2
 - 133s - loss: 0.2264 - mean_squared_error: 0.0513 - val_loss: 0.2273 - val_mean_squared_error: 0.0517
 
 210s - loss: 0.0498 - root_mean_squared_error: 0.1458 - val_loss: 0.0513 - val_root_mean_squared_error: 0.1521
 
lb 0.256
Train on 1353081 samples, validate on 150343 samples
Epoch 1/50

Epoch 00001: val_loss improved from inf to 0.12526, saving model to weights/simpleRNN_attention_v1.h5
 - 277s - loss: 0.1304 - mean_squared_error: 0.0722 - val_loss: 0.1253 - val_mean_squared_error: 0.0676
Epoch 2/50

Epoch 00002: val_loss improved from 0.12526 to 0.12426, saving model to weights/simpleRNN_attention_v1.h5
 - 277s - loss: 0.1261 - mean_squared_error: 0.0673 - val_loss: 0.1243 - val_mean_squared_error: 0.0653
Epoch 3/50

Epoch 00003: val_loss improved from 0.12426 to 0.12376, saving model to weights/simpleRNN_attention_v1.h5
 - 276s - loss: 0.1248 - mean_squared_error: 0.0661 - val_loss: 0.1238 - val_mean_squared_error: 0.0658
Epoch 4/50

Epoch 00004: val_loss improved from 0.12376 to 0.12347, saving model to weights/simpleRNN_attention_v1.h5
 - 275s - loss: 0.1238 - mean_squared_error: 0.0650 - val_loss: 0.1235 - val_mean_squared_error: 0.0644
Epoch 5/50

Epoch 00005: val_loss improved from 0.12347 to 0.12333, saving model to weights/simpleRNN_attention_v1.h5
 - 278s - loss: 0.1230 - mean_squared_error: 0.0645 - val_loss: 0.1233 - val_mean_squared_error: 0.0653
Epoch 6/50

Epoch 00006: val_loss improved from 0.12333 to 0.12316, saving model to weights/simpleRNN_attention_v1.h5
 - 278s - loss: 0.1223 - mean_squared_error: 0.0639 - val_loss: 0.1232 - val_mean_squared_error: 0.0645
Epoch 7/50

Epoch 00007: val_loss improved from 0.12316 to 0.12309, saving model to weights/simpleRNN_attention_v1.h5
 - 276s - loss: 0.1217 - mean_squared_error: 0.0634 - val_loss: 0.1231 - val_mean_squared_error: 0.0642
Epoch 8/50
'''