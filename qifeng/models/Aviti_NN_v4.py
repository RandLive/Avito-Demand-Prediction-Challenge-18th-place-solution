import os; os.environ['OMP_NUM_THREADS'] = '1'
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
    df['price'] = df['price'].fillna(0.)
    df['price'] = y_scaler.fit_transform(np.log1p(df['price'].values.reshape(-1,1)))
    return df[['title', 'text', 'price', 'user_type', 'image_top_1']]

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_test = xs
    gpu_options = tf.GPUOptions(force_gpu_compatible = 8)
    config = tf.ConfigProto(
        log_device_placement=True, allow_soft_placement=True, gpu_options=gpu_options,
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.normalization.BatchNormalization()(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dropout(0.2)(out)
        out = ks.layers.Dense(1, activation='sigmoid')(out)
        model = ks.Model(model_in, out)
        model.compile(loss=root_mean_squared_error, optimizer=ks.optimizers.Adam(lr=3e-3))
        for i in range(5):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=1)
        return model.predict(X_test)[:, 0]
        
def root_mean_squared_error(y_true, y_pred):
    return ks.layers.K.sqrt(ks.layers.K.mean(ks.layers.K.square(y_pred - y_true)))
    
def main():
    vectorizer = make_union(
        on_field('title', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field('text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
        on_field(['price', 'user_type', 'image_top_1'],
                 FunctionTransformer(to_records, validate=False), DictVectorizer()),
        n_jobs=4)
    y_scaler = StandardScaler()
    with timer('process train'):
        train = pd.read_csv('../input/train.csv', parse_dates = ["activation_date"])
        train = train[train['deal_probability'] >= 0].reset_index(drop=True)
        cv = KFold(n_splits=20, shuffle=True, random_state=42)
        train_ids, valid_ids = next(cv.split(train))
        train, valid = train.iloc[train_ids], train.iloc[valid_ids]
        y_train = y_scaler.fit_transform(np.log1p(train['deal_probability'].values.reshape(-1, 1)))
        X_train = vectorizer.fit_transform(preprocess(train)).astype(np.float32)
        print(f'X_train: {X_train.shape} of {X_train.dtype}')
        del train
    with timer('process valid'):
        X_valid = vectorizer.transform(preprocess(valid)).astype(np.float32)
    with ThreadPool(processes=4) as pool:
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        xs = [[Xb_train, Xb_valid], [X_train, X_valid]] * 2
        y_pred = np.mean(pool.map(partial(fit_predict, y_train=y_train), xs), axis=0)
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]
    print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(valid['deal_probability'], y_pred))))

if __name__ == '__main__':
    main()

