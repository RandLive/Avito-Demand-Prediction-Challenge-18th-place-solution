from __future__ import division
import pandas as pd
import numpy as np


# In[2]:

import gc
import subprocess
from ImageDataGenerator import *
import os
import pickle
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, CuDNNGRU, Embedding, Flatten, Activation, BatchNormalization, PReLU
from keras.initializers import he_uniform, RandomNormal
from keras.layers import Conv1D, SpatialDropout1D, Bidirectional, Reshape, Dot, GaussianDropout
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from tqdm import tqdm
from nltk import ngrams
from sklearn.preprocessing import LabelEncoder
from utility import *

from sklearn.metrics import mean_squared_error


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# restrict gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

train_dir = '../input/train_jpg/data/competition_files/train_jpg/'
test_dir = '../input/test_jpg/data/competition_files/test_jpg/'

# test = pd.read_csv('../input/test.csv.zip', parse_dates=["activation_date"])
# train = pd.read_csv('../input/train.csv.zip', parse_dates=["activation_date"])

with open('../input/train_ridge.p', 'rb') as f:
    train = pickle.load(f)
    
with open('../input/test_ridge.p', 'rb') as f:
    test = pickle.load(f)    


gp = pd.read_csv('../input/aggregated_features.csv') 
train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

 

with open('../input/train_image_features.p','rb') as f:
    x = pickle.load(f)
    
train_blurinesses = x['blurinesses']
train_dullnesses = x['dullnesses']
train_whitenesses = x['whitenesses']
train_average_pixel_width = x['average_pixel_width']
train_widths = x['widths']
train_heights = x['heights']
train_ids = x['ids']

del x
gc.collect()

with open('../input/test_image_features.p','rb') as f:
    x = pickle.load(f)

test_blurinesses = x['blurinesses']
test_dullnesses = x['dullnesses']
test_whitenesses = x['whitenesses']
test_average_pixel_width = x['average_pixel_width']
test_widths = x['widths']
test_heights = x['heights']
test_ids = x['ids']    


del x
gc.collect()



incep_train_image_df = pd.DataFrame(train_blurinesses, columns = ['blurinesses'])
incep_test_image_df = pd.DataFrame(test_blurinesses, columns = [f'blurinesses'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')  

incep_train_image_df = pd.DataFrame(train_dullnesses, columns = ['dullnesses'])
incep_test_image_df = pd.DataFrame(test_dullnesses, columns = [f'dullnesses'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

incep_train_image_df = pd.DataFrame(train_whitenesses, columns = ['whitenesses'])
incep_test_image_df = pd.DataFrame(test_whitenesses, columns = [f'whitenesses'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

incep_train_image_df = pd.DataFrame(train_widths, columns = ['widths'])
incep_test_image_df = pd.DataFrame(test_widths, columns = [f'widths'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

incep_train_image_df = pd.DataFrame(train_heights, columns = ['heights'])
incep_test_image_df = pd.DataFrame(test_heights, columns = [f'heights'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

with open('../input/train_image_features_1.p','rb') as f:
    x = pickle.load(f)
    
train_average_pixel_width = x['average_pixel_width']
train_average_reds = x['average_reds']
train_average_greens = x['average_greens']
train_average_blues = x['average_blues']
train_ids = x['ids']

del x
gc.collect()

with open('../input/test_image_features_1.p','rb') as f:
    x = pickle.load(f)

test_average_pixel_width = x['average_pixel_width']
test_average_reds = x['average_reds']
test_average_greens = x['average_greens']
test_average_blues = x['average_blues']
test_ids = x['ids']  

del x
gc.collect()

incep_train_image_df = pd.DataFrame(train_average_pixel_width, columns = ['average_pixel_width'])
incep_test_image_df = pd.DataFrame(test_average_pixel_width, columns = [f'average_pixel_width'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

incep_train_image_df = pd.DataFrame(train_average_reds, columns = ['average_reds'])
incep_test_image_df = pd.DataFrame(test_average_reds, columns = [f'average_reds'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

incep_train_image_df = pd.DataFrame(train_average_blues, columns = ['average_blues'])
incep_test_image_df = pd.DataFrame(test_average_blues, columns = [f'average_blues'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')


incep_train_image_df = pd.DataFrame(train_average_greens, columns = ['average_greens'])
incep_test_image_df = pd.DataFrame(test_average_greens, columns = [f'average_greens'])
incep_train_image_df['image'] = train_ids
incep_test_image_df['image'] = test_ids
train = train.join(incep_train_image_df.set_index('image'), on='image')
test = test.join(incep_test_image_df.set_index('image'), on='image')

train_image_feat = pd.read_csv('../input/train_image_feature_new.csv')
test_image_feat = pd.read_csv('../input/test_image_feature_new.csv')

new_image_feat = list(set(train_image_feat.columns) - set(['image']))

new_image_feat = [f for f in new_image_feat]

train = train.merge(train_image_feat, on='image', how='left')
test = test.merge(test_image_feat, on='image', how='left')

train_image_feat = pd.read_csv('../input/train_image_feature_newer.csv')
test_image_feat = pd.read_csv('../input/test_image_feature_newer.csv')

newer_image_feat = list(set(train_image_feat.columns) - set(['image']))

#LAB, YCrCb
newer_image_feat = [f for f in newer_image_feat if 'YCrCb' in f]

train = train.merge(train_image_feat, on='image', how='left')
test = test.merge(test_image_feat, on='image', how='left')

data = pd.concat([train, test], axis=0, sort=False)

with open('../input/region_income.csv', 'r')as f:
    region_incomes = [i.strip() for i in f.readlines()]
    
region_income_map = {}
for line in region_incomes:
    region, income = line.split(';')
    region_income_map[region] = int(income)

data['income'] = data.region.map(region_income_map )

    
text_col = ['title', 'description', 'param_1', 'param_2', 'param_3']
for c in text_col:
    data[c].fillna(f'No {c}', inplace=True)
    train[c].fillna(f'No {c}', inplace=True)

# mean_image_quality = data.loc[~data.image_quality.isna(), 'image_quality'].mean()
# data['image_quality'].fillna(mean_image_quality, inplace=True)    

mean_blurinesses= data.loc[~data.blurinesses.isna(), 'blurinesses'].mean()
data['blurinesses'].fillna(mean_blurinesses, inplace=True)   

mean_dullnesses= data.loc[~data.dullnesses.isna(), 'dullnesses'].mean()
data['dullnesses'].fillna(mean_dullnesses, inplace=True)

mean_whitenesses= data.loc[~data.whitenesses.isna(), 'whitenesses'].mean()
data['whitenesses'].fillna(mean_whitenesses, inplace=True)

for f in new_image_feat:
    mean_val = data.loc[~data[f].isna(), f].mean()
    data[f].fillna(0, inplace=True)

for f in newer_image_feat:
    mean_val = data.loc[~data[f].isna(), f].mean()
    data[f].fillna(0, inplace=True)    
    data[f] = np.log1p(data[f]).astype(np.float32)
# mean_widths= data.loc[~data.widths.isna(), 'widths'].mean()
# data['widths'].fillna(mean_widths, inplace=True)

# mean_heights= data.loc[~data.heights.isna(), 'heights'].mean()
# data['heights'].fillna(mean_heights, inplace=True)


data['average_pixel_width'].fillna(0, inplace=True)
data['average_reds'].fillna(0, inplace=True)
data['average_greens'].fillna(0, inplace=True)
data['average_blues'].fillna(0, inplace=True)
# data['widths'].fillna(0, inplace=True)
# data['heights'].fillna(0, inplace=True)

city_population = pd.read_csv('../input/city_population_wiki_v3.csv')
data = data.merge(city_population, on='city', how='left')

mean_population = city_population.population.mean()
data['population'].fillna(mean_population, inplace=True)

data['des_len'] = data.description.str.len()
data['des_nwords'] = data.description.str.split().apply(len)  
train['des_len'] = train.description.str.len()
train['des_nwords'] = train.description.str.split().apply(len)  


data['des_len_log'] = (np.log(data['des_len']) * 4).astype(np.int8)
data['des_nwords_log'] = (np.log1p(data['des_nwords']) * 20).astype(np.int8)
train['des_len_log'] = (np.log(train['des_len']) * 4).astype(np.int8)
train['des_nwords_log'] = (np.log1p(train['des_nwords']) * 20).astype(np.int8)

mean_price = data.loc[~data.price.isna(), 'price'].mean()
data['price'].fillna(mean_price, inplace=True)
train['price'].fillna(mean_price, inplace=True)

#log scale
data['item_seq_number'] = np.log1p(data['item_seq_number']).astype(np.float32)
data['price'] = np.log1p(data['price']).astype(np.float32)
data['population'] = np.log1p(data['population']).astype(np.float32)
data['income'] = np.log1p(data['income']).astype(np.float32)
data['blurinesses'] = np.log1p(data['blurinesses']).astype(np.float32)
data['whitenesses'] = np.log1p(data['whitenesses']).astype(np.float32)
data['dullnesses'] = np.log1p(data['dullnesses']).astype(np.float32)

data["price_p"] = np.round(data["price"] * 4).astype(np.int16)
train["price_p"] = np.round(train["price"] * 4).astype(np.int16)



cont_features = ['price']
cont_features.append('des_len')
cont_features.append('des_nwords')
cont_features.append('item_seq_number')
cont_features.append('ridge_feature')
cont_features.append('avg_days_up_user')
cont_features.append('avg_times_up_user')
cont_features.append('n_user_items')
cont_features.append('population')
cont_features.append('income')

cont_features.append('whitenesses')
cont_features.append('dullnesses')
cont_features.append('blurinesses')
cont_features.append('average_pixel_width')
cont_features.append('average_reds')
cont_features.append('average_greens')
cont_features.append('average_blues')
# cont_features.append('widths')
# cont_features.append('heights')
cont_features.extend(new_image_feat)
# cont_features.extend(newer_image_feat)



data["item_seq_number_cat"] = np.round(data["item_seq_number"] * 8).astype(int) 
train["item_seq_number_cat"] = np.round(train["item_seq_number"] * 8).astype(int) 

agg_cols = ['param_1', 'category_name', 'region', 'user_type', 'price_p', 'item_seq_number_cat', 'city', 'parent_category_name', 'des_len_log', 'des_nwords_log']

for c in tqdm(agg_cols):
    gp = train.groupby(c)['deal_probability']
    mean = gp.mean()
    data[str(c) + '_deal_probability_avg'] = data[c].map(mean)
    cont_features.append(str(c) + '_deal_probability_avg')    

data.fillna(0, inplace=True)     

# normalizing
for c in tqdm(new_image_feat ):
    data[c] = MinMaxScaler().fit_transform(data[c].values.reshape(-1,1))    

# normalizing
# for c in tqdm(newer_image_feat):
#     data[c] = MinMaxScaler().fit_transform(data[c].values.reshape(-1,1))        
    
cate_cols = ['city',  'category_name', 'user_type','parent_category_name','region','param_1','param_2','param_3', 'image_top_1', 'price_p', 'item_seq_number_cat']

for c in tqdm(cate_cols):    
    data[c] = LabelEncoder().fit_transform(data[c].values.astype('str'))    



new_data = data.drop(['user_id','description','title'], axis=1)
# new_data.fillna(0, inplace= True)



x_train = new_data.loc[new_data.activation_date<=pd.to_datetime('2017-04-07')]
x_test = new_data.loc[new_data.activation_date>=pd.to_datetime('2017-04-08')]

# the word must appear at least five time        


with open(f'../input/train_des_seq.p','rb') as f:
    train_des_seq = pickle.load(f)
    
with open(f'../input/test_des_seq.p','rb') as f:
    test_des_seq = pickle.load(f)
    
with open(f'../input/train_title_seq.p','rb') as f:
    train_title_seq = pickle.load(f)    

with open(f'../input/test_title_seq.p','rb') as f:
    test_title_seq = pickle.load(f)
    


print(f'cont_features: {cont_features}', flush=True)
print(f'columns: {new_data.columns}', flush=True)
     


with open(f'../input/word_index.p','rb') as f:
    word_index = pickle.load(f)    

max_text = len(word_index)+1
# with open(f'../input/embedding_matrix_pk.p', 'rb') as f:
#     embedding_matrix =  pickle.load(f)       




max_des_len = 80
max_title_len = 30


max_user_type= np.max(new_data['user_type'].max()) + 1
max_parent_category_name= np.max(new_data['parent_category_name'].max()) + 1
max_category_name= np.max(new_data['category_name'].max()) + 1
max_param_1 = np.max(new_data['param_1'].max()) + 1
max_param_2 = np.max(new_data['param_2'].max()) + 1
max_param_3 = np.max(new_data['param_3'].max()) + 1
max_region = np.max(new_data['region'].max()) + 1
max_city = np.max(new_data['city'].max()) + 1
max_image_top_1 = np.max(new_data['image_top_1'].max()) + 1
max_price_p = np.max(new_data['price_p'].max())+1
max_item_seq_number_cat = np.max(new_data['item_seq_number_cat'].max()) + 1
# max_lat_lon_hdbscan_cluster_20_03 = np.max(new_data['lat_lon_hdbscan_cluster_20_03'].max())+1
# max_lat_lon_hdbscan_cluster_10_03 = np.max(new_data['lat_lon_hdbscan_cluster_10_03'].max())+1
# max_lat_lon_hdbscan_cluster_05_03 = np.max(new_data['lat_lon_hdbscan_cluster_05_03'].max())+1
# max_param_123 = np.max(new_data['param_123'].max()) + 1

del data, new_data
gc.collect()
y_train = x_train['deal_probability'].values
x_train = x_train.drop(['deal_probability','activation_date'],axis=1)
x_test = x_test.drop(['deal_probability','activation_date'],axis=1)

# x_train = get_keras_fasttext(X, train_des_seq, train_title_seq)
# x_test = get_keras_fasttext(X_te, test_des_seq, test_title_seq)
# print(f"cont size: {x_train['cont_features'].shape}")    

hyper_params={
    'description_filters':40,
    'embedding_dim':80,
    'enable_deep':False,
    'enable_fm':False,
    'learning_rate':0.0001
    
}
print(hyper_params, flush=True)
def gauss_init():
    return RandomNormal(mean=0.0, stddev=0.005)

def get_model():
    description = Input(shape=[train_des_seq.shape[1]], name="description")
    title = Input(shape=[train_title_seq.shape[1]], name="title")
    user_type = Input(shape=[1], name="user_type")
    category_name = Input(shape=[1], name="category_name")
    parent_category_name = Input(shape=[1], name="parent_category_name")    
    param_1 = Input(shape=[1], name="param_1")
    param_2 = Input(shape=[1], name="param_2")
    param_3 = Input(shape=[1], name="param_3")
    region = Input(shape=[1], name="region")
    city = Input(shape=[1], name="city")
    image_top_1 = Input(shape=[1], name="image_top_1")
    price_p = Input(shape=[1], name='price_p')
    item_seq_number_cat = Input(shape=[1], name='item_seq_number_cat')
    image = Input (shape=(*dim, 3), name='image')
#     image_x = Conv2D(32, (3,3) , activation='relu')(image)
#     image_x = MaxPooling2D((2, 2))(image_x)
#     image_x = Conv2D(64, (3,3) , activation='relu')(image_x)
#     image_x = MaxPooling2D((2, 2))(image_x)
#     image_x = Conv2D(64, (3,3) , activation='relu')(image_x)
#     image_x = MaxPooling2D((2, 2))(image_x)
#     image_x = Conv2D(64, (3,3) , activation='relu')(image_x)
#     image_x = MaxPooling2D((2, 2))(image_x)
#     image_x = Conv2D(64, (3,3) , activation='relu')(image_x)
#     image_x = MaxPooling2D((2, 2))(image_x)
#     image_x = Flatten()(image_x)
    x = conv2d_bn(image, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    image_x = inceptionv3_block(x, 'incep1')
#     image_x = inceptionv3_block(image_x, 'incep2')    
    image_x = GlobalAveragePooling2D()(image_x)

#     image_x = Dense(80)(image_x)
    
#     cat_ipnuts = [Input(shape=[1]) for feat in cat_cols]
    
    continuous_inputs = [Input(shape=[1], name=feat) for feat in cont_features]
      
    shared_embedding = Embedding(max_text, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())    

    emb_description = shared_embedding (description)    

    emb_title = shared_embedding (title)

    emb_user_type =  ( Embedding(max_user_type, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(user_type)    )
    emb_param_1 =  ( Embedding(max_param_1, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_1) )
    emb_param_2 =  ( Embedding(max_param_2, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_2) )
    emb_param_3 = ( Embedding(max_param_3, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(param_3) )
    emb_category_name =   ( Embedding(max_category_name, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(category_name) )
    emb_parent_category_name =   ( Embedding(max_parent_category_name, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(parent_category_name) )
    emb_region = ( Embedding(max_region, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(region) )
    emb_city = ( Embedding(max_city, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(city) )
    emb_image_top_1 =  ( Embedding(max_image_top_1, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(image_top_1) )
    emb_price_p = ( Embedding(max_price_p, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(price_p) )
    emb_item_seq_number_cat = ( Embedding(max_item_seq_number_cat, hyper_params['embedding_dim'], embeddings_initializer = gauss_init())(item_seq_number_cat) )
    
    
    att_description = GlobalMaxPooling1D( name='output_des_max' )(emb_description)
    att_title = GlobalMaxPooling1D(name='output_title_max' )(emb_title)

    cont_denses = [ Dense(hyper_params['embedding_dim'])(c) for c in continuous_inputs]
    common_list = [
                    Flatten()(emb_image_top_1),
                    Flatten()(emb_region),
                    Flatten()(emb_city),
                    Flatten()(emb_param_1),
                    Flatten()(emb_param_2),
                    Flatten()(emb_param_3),
                    Flatten()(emb_user_type),
                    Flatten()(emb_price_p),        
                    *cont_denses,
                    ]
    fm_list = []
    if hyper_params['enable_fm']:

        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                inner_product = Dot(axes=1)([common_list[i], common_list[j]])
                fm_list.append(inner_product)
   

    final_list = [ att_description, 
                    att_title, 
                    # emb_gru,
                    Flatten()  (emb_region), 
                    Flatten()   (emb_city), 
                    Flatten() ( emb_category_name), 
                    Flatten()  (emb_parent_category_name),
                    Flatten() (emb_user_type), 
                    Flatten()  (emb_param_1), 
                    Flatten()  (emb_param_2), 
                    Flatten()   (emb_param_3), 
                    Flatten()  (emb_image_top_1),  
                    Flatten()  (emb_price_p),  
                    Flatten()  (emb_item_seq_number_cat),                    
                    *continuous_inputs]

    
    x = concatenate(final_list )
    if hyper_params['enable_fm']:
        x = concatenate([x, *fm_list])
    x = BatchNormalization()(x)
    
    x = Dense(512)(x)

    x = Activation('relu')(x)
    x = concatenate([x, image_x])
    x = Dense(64)(x)
    
    x = Activation('relu')(x)

    x = Dense(1, activation="sigmoid") (x)
    model = Model([description, title,  user_type , category_name, parent_category_name, param_1, param_2, param_3,
                   region,city, image_top_1, price_p, item_seq_number_cat, image, *continuous_inputs] ,
                   x)
    optimizer = Adam(hyper_params['learning_rate'], amsgrad=True)
#     optimizer = Nadam(hyper_params['learning_rate'])
    model.compile(loss="mse", optimizer=optimizer)
    return model




def train_bagging(X, y, fold_count, des_seq, title_seq):
    
    
    kf = KFold(n_splits=fold_count, random_state=42, shuffle=True)
#     skf = StratifiedKFold(n_splits=fold_count, random_state=None, shuffle=False)
    fold_id = -1
#     model_list = []
    val_predict= np.zeros(y.shape)
#     rmse_list = []
    for train_index, test_index in kf.split(y):
        
        fold_id +=1 
        if fold_id >= 1: exit()

        print(f'fold number: {fold_id}', flush=True)
        
        

        x_train, x_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        
        x_train, x_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y[train_index], y[test_index]
        
        train_des_seq, val_des_seq = des_seq[train_index], des_seq[test_index]
        train_title_seq, val_title_seq = title_seq[train_index], title_seq[test_index]
        
        x_train.set_index('item_id', inplace=True)
        x_val.set_index('item_id', inplace=True)        
        
        train_item_ids = x_train.index
        val_item_ids =  x_val.index
        
        train_image_ids = x_train.image
        val_image_ids = x_val.image
        
        train_labels = y_train
        val_labels = y_val
        
        
        
#         print(len(val_labels))
        
        train_gen = ImageDataGenerator(train_dir, train_item_ids, train_image_ids, train_labels, dim= dim, df = x_train, title_seq = train_title_seq, des_seq = train_des_seq, cont_features=cont_features)
        
        val_gen = ImageDataGenerator(train_dir, val_item_ids, val_image_ids, val_labels, dim = dim, shuffle=False, df=x_val, title_seq = val_title_seq, des_seq = val_des_seq, cont_features=cont_features)
        
        
        model_path = f'../weights/{fname}_fold{fold_id}.hdf5'
        #if model weights exist
        if os.path.exists(model_path):
            print('weight loaded')
            model = get_model()
            model.load_weights(model_path, by_name=True)
            y_pred = model.predict_generator(val_gen, verbose=1)      
            val_predict[test_index] = y_pred[:,0]
#             model_list.append(model)
            rmse = mean_squared_error(y_val, y_pred) ** 0.5
            

            del model
            gc.collect()
            print(f'rmse: {rmse}')
            rmse_list.append(rmse)
            continue
        
        model = get_model()
        
        

        early= EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
#         rlrop = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=2,verbose=1,factor=0.1,cooldown=0,min_lr=1e-6)
        callbacks = [early, checkpoint]
        
        model.fit_generator(train_gen,  validation_data=val_gen, callbacks=callbacks, epochs=epochs, verbose=1, use_multiprocessing=True, workers=workers)
        model.load_weights(model_path, by_name=True)
        y_pred = model.predict_generator(val_gen, verbose=1,use_multiprocessing=True, workers=workers)         
        val_predict[test_index] = y_pred[:,0]
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
#         train_rmse = mean_squared_error(model.predict(x_train), y_train) ** 0.5
#         print(f'train_rmse {train_rmse}')
        print(f'rmse: {rmse}')
        y_pred = model.predict_generator(test_gen, verbose=1,use_multiprocessing=True, workers=workers)
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['deal_probability'] = y_pred
        sub['deal_probability'].clip(0.0, 1.0, inplace=True)
        sub.to_csv(f'../output/{fname}_test_fold{fold_id}.csv', index=False)
        del model
        gc.collect()
        rmse_list.append(rmse)
#         model_list.append(model)
    print(f'rmse score avg: {np.mean(rmse_list)}', flush=True)
    return val_predict




# In[57]:


# weight_dir = '../weights/'
epochs = 10
batch_size = 128
nfold = 10
rmse_list = []
dim = (200, 200)
workers=4

fname = 'emb_all_80_itseqcat_price_p4_pr1_catn_rg_ut_p_ict_pc_dll_dnl_rgb_apw_newmm_b128_img_dim200_incepv3_avgpool_imagemiddle_10fold'


print(get_model().summary(), flush=True)
print(f'fname {fname}', flush=True)




x_test.set_index('item_id', inplace=True)


test_item_ids = x_test.index


test_image_ids = x_test.image


        
        
        
#         print(len(val_labels))
        
test_gen = ImageDataGenerator(test_dir, test_item_ids, test_image_ids, None, dim = dim, is_train=False, shuffle=False, df=x_test, title_seq = test_title_seq, des_seq = test_des_seq, cont_features=cont_features )

val_predict = train_bagging(x_train, y_train, nfold, train_des_seq, train_title_seq)

model = get_model()

print('storing test prediction', flush=True)
for index in tqdm(range(nfold)):
    model_path = f'../weights/{fname}_fold{index}.hdf5'
    model.load_weights(model_path)
    if index == 0: 
        y_pred = model.predict_generator(test_gen,use_multiprocessing=True, workers=workers, verbose=1)
    else:
        y_pred *= model.predict_generator(test_gen,use_multiprocessing=True, workers=workers, verbose=1)

    
y_pred = np.clip(y_pred, 0, 1)
y_pred = y_pred **( 1.0/ (nfold))


print('storing test prediction', flush=True)
sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv(f'../output/{fname}_test.csv', index=False)


print('storing oof prediction', flush=True)
train_data = pd.read_csv('../input/train.csv.zip')
label = ['deal_probability']
train_user_ids = train_data.user_id.values
train_item_ids = train_data.item_id.values

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_user_ids.reshape(len(train_user_ids), 1)

val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv(f'../output/{fname}_train.csv', index=False)
