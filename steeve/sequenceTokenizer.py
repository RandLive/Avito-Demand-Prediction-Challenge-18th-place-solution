import pandas as pd
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
from keras.preprocessing.text import Tokenizer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# restrict gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.preprocessing.sequence import pad_sequences
test = pd.read_csv('../input/test.csv.zip')
train = pd.read_csv('../input/train.csv.zip')

stopWords = stopwords.words('russian')
print('to lower')


# train['text_feat'] = train.apply(lambda row: ' '.join([
#     str(row['param_1']), 
#     str(row['param_2']), 
#     str(row['param_3'])]),axis=1) # Group Param Features

# test['text_feat'] = test.apply(lambda row: ' '.join([
#     str(row['param_1']), 
#     str(row['param_2']), 
#     str(row['param_3'])]),axis=1) # Group Param Features

# train['text_feat'].fillna('', inplace=True)
# test['text_feat'].fillna('', inplace=True)
# train_des = train.text_feat.str.lower()
# test_des = test.text_feat.str.lower()


train['description'].fillna('', inplace=True)
train_des = train.description
test_des = test.description

train['title'].fillna('', inplace=True)
test['title'].fillna('', inplace=True)
train_title = train.title
test_title = test.title



# remove speical character
print('removing characters')

cleaned_train_des = [re.sub('\\s+', ' ', des) for des in train_des.values]
cleaned_test_des = [re.sub('\\s+', ' ', des)  for des in test_des.values]
cleaned_train_title = [re.sub('\\s+', ' ', des)  for des in train_title.values]
cleaned_test_title = [re.sub('\\s+', ' ', des) for des in test_title.values]


# remove number
print('removing numbers')
cleaned_train_des = [re.sub('[%s|0123456789]' % re.escape(string.punctuation), ' ', des) for des in cleaned_train_des]
cleaned_test_des = [re.sub('[%s|0123456789]' % re.escape(string.punctuation), ' ', des) for des in cleaned_test_des]
cleaned_train_title = [re.sub('[%s|0123456789]' % re.escape(string.punctuation), ' ', des) for des in cleaned_train_title]
cleaned_test_title = [re.sub('[%s|0123456789]' % re.escape(string.punctuation), ' ', des) for des in cleaned_test_title]
# remove alpha?
print('removing alphas')
cleaned_train_des = [re.sub('[^[:alpha:]]', ' ', des) for des in cleaned_train_des]
cleaned_test_des = [re.sub('[^[:alpha:]]', ' ', des) for des in cleaned_test_des]
cleaned_train_title = [re.sub('[^[:alpha:]]', ' ', des) for des in cleaned_train_title]
cleaned_test_title = [re.sub('[^[:alpha:]]', ' ', des) for des in cleaned_test_title]



min_df_one=5        

# def word_count(text, dc):
#     text = set( text.split() ) 
#     for w in text:
#         dc[w]+=1

# def remove_low_freq(text, dc):
#     return ' '.join( [w for w in text.split() if w in dc] )

print('building vocab')
num_words=max_words=150000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(cleaned_test_des + cleaned_train_des + cleaned_train_title + cleaned_test_title)

# word_count_dict_one = defaultdict(np.uint32)        
# for des in cleaned_test_des + cleaned_train_des + cleaned_train_title + cleaned_test_title:
#     word_count(des, word_count_dict_one)        

# rare_words = [key for key in word_count_dict_one if  word_count_dict_one[key]<min_df_one ]
# for key in rare_words :
#     word_count_dict_one.pop(key, None)
    
# for i in cleaned_train_des:
#     cleaned_train_des[i] = remove_low_freq(cleaned_train_des, word_count_dict_one)
# word_count_dict_one=dict(word_count_dict_one)
    
# with open('../input/vocab_one.p','wb') as f:
#     pickle.dump(word_count_dict_one, f, protocol=4)
word_index = tokenizer.word_index

with open(f'../input/word_index{num_words}.p','wb') as f:
    pickle.dump(word_index, f, protocol=4)


print(len(word_index))

print('padding')    
max_des_len = 80
max_title_len = 30    
def preprocess_keras(text):
    return [ word_count_dict_one[w] for w in (text.split())[:max_des_len] ]


# cleaned_train_des = [preprocess_keras(des) for des in cleaned_train_des]
# cleaned_test_des = [preprocess_keras(des) for des in cleaned_test_des]
# cleaned_train_title = [preprocess_keras(des) for des in cleaned_train_title]
# cleaned_test_title = [preprocess_keras(des) for des in cleaned_test_title]



cleaned_train_des = tokenizer.texts_to_sequences(cleaned_train_des)
cleaned_test_des  = tokenizer.texts_to_sequences(cleaned_test_des)
cleaned_train_title= tokenizer.texts_to_sequences(cleaned_train_title)
cleaned_test_title = tokenizer.texts_to_sequences(cleaned_test_title)

train_des_seq = pad_sequences(cleaned_train_des, maxlen=max_des_len)
test_des_seq = pad_sequences(cleaned_test_des, maxlen=max_des_len)
train_title_seq = pad_sequences(cleaned_train_title, maxlen=max_title_len)
test_title_seq = pad_sequences(cleaned_test_title, maxlen=max_title_len)


print(train_des_seq.max().max())
print(train_des_seq.max().max())
print(train_des_seq.max().max())
print(train_des_seq.max().max())
with open(f'../input/train_des_seq{num_words}.p','wb') as f:
    pickle.dump(train_des_seq, f, protocol=4)
    
with open(f'../input/test_des_seq{num_words}.p','wb') as f:
    pickle.dump(test_des_seq, f, protocol=4)
    
with open(f'../input/train_title_seq{num_words}.p','wb') as f:
    pickle.dump(train_title_seq, f, protocol=4)
    
with open(f'../input/test_title_seq{num_words}.p','wb') as f:
    pickle.dump(test_title_seq, f, protocol=4)    
