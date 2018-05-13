# import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import gc

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    y_scaler = StandardScaler()
    df['title'] = df['title'].fillna('')
    df['user_type'] = df['user_type'].fillna('')
    df['image_top_1'] = df['title'].fillna(-1)
    df['desdescription'] = df['description'].fillna('')
    df['text'] = (df['description'].fillna('') + ' ' + df['title'])
    df['price'] = df['price'].fillna(-1000)
    df['price'] = y_scaler.fit_transform(df['price'].values.reshape(-1,1))
    return df[['title', 'text', 'price', 'user_type', 'image_top_1']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.normalization.BatchNormalization()(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1, activation='sigmoid')(out)
        model = ks.Model(model_in, out)
        model.compile(loss=root_mean_squared_error, optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        return model.predict(X_test)[:, 0]
        
def root_mean_squared_error(y_true, y_pred):
    return ks.layers.K.sqrt(ks.layers.K.mean(ks.layers.K.square(y_pred - y_true)))
    
def main():
    vectorizer = make_union(
        on_field('title', Tfidf(max_features=100000, token_pattern='\w+')),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['price', 'user_type', 'image_top_1'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=8)
    y_scaler = StandardScaler()
    
    with timer('process train'):
        print('read train data ...')
        train = pd.read_csv('../input/train.csv', parse_dates = ["activation_date"])


#        cv = KFold(n_splits=10, shuffle=True, random_state=42)
#        train_ids, valid_ids = next(cv.split(train))
#        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        
        train, valid = train_test_split(train, test_size=0.10, random_state=23)
        y_train = y_scaler.fit_transform(train['deal_probability'].values.reshape(-1, 1))
        X_train = vectorizer.fit_transform(preprocess(train))       
        print('X_train: {} of {}'.format(X_train.shape,X_train.dtype))
        del train; gc.collect()
        
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid))
        
        
    with timer('process test'):
        # TODO
        print('read test data ...')
        test = pd.read_csv('../input/test.csv', parse_dates = ["activation_date"])
        X_test = vectorizer.transform(preprocess(test))
        del test; gc.collect()

     
    with ThreadPool(processes=8) as pool:
#        Xb_train, Xb_valid = [x.astype(np.bool) for x in [X_train, X_valid]]
        Xb_train, Xb_valid, Xb_test = [x.astype(np.bool) for x in [X_train, X_valid, X_test]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        del X_valid; gc.collect()
        
        # TODO
        xs_test = [[Xb_train, Xb_test], [X_train, Xb_test]] * 2
        del X_train, X_test; gc.collect()
    
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)
        
        # TODO
        y_pred_test = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs_test), axis=0)
        
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]
    
    # TODO
    y_pred_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1))[:, 0]
    
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['deal_probability'], y_pred))))
    del valid; gc.collect()
    
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['deal_probability'] = y_pred_test
    sub.to_csv('sub.csv', index=False)
    print('all done!')

if __name__ == '__main__':
    main()