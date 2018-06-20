import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
from utils import *
import argparse
from sklearn import preprocessing
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

class TargetEncoder:
    # Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
    def __repr__(self):
        return 'TargetEncoder'

    def __init__(self, cols, smoothing=1, min_samples_leaf=1, noise_level=0, keep_original=False):
        self.cols = cols
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.keep_original = keep_original

    @staticmethod
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def encode(self, train, test, target):
        for col in self.cols:
            if self.keep_original:
                train[col + '_te'], test[col + '_te'] = self.encode_column(train[col], test[col], target)
            else:
                train[col], test[col] = self.encode_column(train[col], test[col], target)
        return train, test

    def encode_column(self, trn_series, tst_series, target):
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - self.min_samples_leaf) / self.smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(['mean', 'count'], axis=1, inplace=True)
        # Apply averages to trn and tst series
        ft_trn_series = pd.merge(
            trn_series.to_frame(trn_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=trn_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_trn_series.index = trn_series.index
        ft_tst_series = pd.merge(
            tst_series.to_frame(tst_series.name),
            averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
            on=tst_series.name,
            how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
        # pd.merge does not keep the index so restore it
        ft_tst_series.index = tst_series.index
        return self.add_noise(ft_trn_series, self.noise_level), self.add_noise(ft_tst_series, self.noise_level)

def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

stopwords = {x: 1 for x in stopwords.words('russian')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])


train = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
test = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])

import string

def normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stopwords)

print(train.description[0])
print(normalize_text(train.description[0]))


train['is_train'] = 1
test['is_train'] = 0
print('[{}] Compiled train / test'.format(time.time() - start_time))
print('Train shape: ', train.shape)
print('Test shape: ', test.shape)

y = train.deal_probability.copy()
nrow_train = train.shape[0]

merge = pd.concat([train, test])
submission = pd.DataFrame(test.index)
print('[{}] Compiled merge'.format(time.time() - start_time))
print('Merge shape: ', merge.shape)

del train
del test
gc.collect()
print('[{}] Garbage collection'.format(time.time() - start_time))

print("Feature Engineering - Part 1")
merge["price"] = np.log(merge["price"]+0.001)
merge["price"].fillna(-999,inplace=True)
merge["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
merge["activation_weekday"] = merge['activation_date'].dt.weekday
print(merge.head(5))
gc.collect()

training_index = merge.loc[merge.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = merge.loc[merge.activation_date>=pd.to_datetime('2017-04-08')].index
merge.drop(["activation_date","image"],axis=1,inplace=True)

#Drop user_id
merge.drop(["user_id"], axis=1,inplace=True)

print("\nText Features")
textfeats = ["description", "title"]

for cols in textfeats:
    merge[cols] = merge[cols].astype(str)
    merge[cols] = merge[cols].astype(str).fillna('missing') # FILL NA
    merge[cols] = merge[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    merge[cols + '_num_stopwords'] = merge[cols].apply(lambda x: len([w for w in x.split() if w in stopwords])) # Count number of Stopwords
    merge[cols + '_num_punctuations'] = merge[cols].apply(lambda comment: (comment.count(RE_PUNCTUATION))) # Count number of Punctuations
    merge[cols + '_num_alphabets'] = merge[cols].apply(lambda comment: len([c for c in comment if c.isupper()])) # Count number of Alphabets
    merge[cols + '_num_digits'] = merge[cols].apply(lambda comment: (comment.count('[0-9]'))) # Count number of Digits
    merge[cols + '_num_letters'] = merge[cols].apply(lambda comment: len(comment)) # Count number of Letters
    merge[cols + '_num_words'] = merge[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    merge[cols + '_num_unique_words'] = merge[cols].apply(lambda comment: len(set(w for w in comment.split())))
    merge[cols + '_words_vs_unique'] = merge[cols+'_num_unique_words'] / merge[cols+'_num_words'] # Count Unique Words
    merge[cols + '_letters_per_word'] = merge[cols+'_num_letters'] / merge[cols+'_num_words'] # Letters per Word
    merge[cols + '_punctuations_by_letters'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_letters'] # Punctuations by Letters
    merge[cols + '_punctuations_by_words'] = merge[cols+'_num_punctuations'] / merge[cols+'_num_words'] # Punctuations by Words
    merge[cols + '_digits_by_letters'] = merge[cols+'_num_digits'] / merge[cols+'_num_letters'] # Digits by Letters
    merge[cols + '_alphabets_by_letters'] = merge[cols+'_num_alphabets'] / merge[cols+'_num_letters'] # Alphabets by Letters
    merge[cols + '_stopwords_by_words'] = merge[cols+'_num_stopwords'] / merge[cols+'_num_words'] # Stopwords by Letters
    merge[cols + '_mean'] = merge[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10 # Mean

# Extra Feature Engineering
merge['title_desc_len_ratio'] = merge['title_num_letters']/merge['description_num_letters']

df_test = merge.loc[merge['is_train'] == 0]
df_train = merge.loc[merge['is_train'] == 1]
del merge
gc.collect()
df_test = df_test.drop(['is_train'], axis=1)
df_train = df_train.drop(['is_train'], axis=1)

print(df_train.shape)
print(y.shape)

if SUBMIT_MODE:
    y_train = y
    del y
    gc.collect()
else:
    df_train, df_test, y_train, y_test = train_test_split(df_train, y, test_size=0.2, random_state=144)

print('[{}] Splitting completed.'.format(time.time() - start_time))

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29,
                                                              "norm": None,
                                                              "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze = True
X_name_train = wb.fit_transform(df_train['title'])
print(X_name_train.shape)
X_name_test = wb.transform(df_test['title'])
print(X_name_test.shape)
del(wb)
gc.collect()

mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]
X_name_train = X_name_train[:, mask]
print(X_name_train.shape)
X_name_test = X_name_test[:, mask]
print(X_name_test.shape)
print('[{}] Vectorize `title` completed.'.format(time.time() - start_time))

X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train, y_train,
                                                              test_size = 0.5,
                                                              shuffle = False)
print('[{}] Finished splitting'.format(time.time() - start_time))

model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
model.fit(X_train_1, y_train_1)
print('[{}] Finished to train name ridge (1)'.format(time.time() - start_time))
name_ridge_preds1 = model.predict(X_train_2)
name_ridge_preds1f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (1)'.format(time.time() - start_time))
model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
model.fit(X_train_2, y_train_2)
print('[{}] Finished to train name ridge (2)'.format(time.time() - start_time))
name_ridge_preds2 = model.predict(X_train_1)
name_ridge_preds2f = model.predict(X_name_test)
print('[{}] Finished to predict name ridge (2)'.format(time.time() - start_time))
name_ridge_preds_oof = np.concatenate((name_ridge_preds2, name_ridge_preds1), axis=0)
name_ridge_preds_test = (name_ridge_preds1f + name_ridge_preds2f) / 2.0
print('RMSLE OOF: {}'.format(rmse(name_ridge_preds_oof, y_train)))
if not SUBMIT_MODE:
    print('RMSLE TEST: {}'.format(rmse(name_ridge_preds_test, y_test)))
gc.collect()

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                              "hash_ngrams_weights": [1.0, 1.0],
                                                              "hash_size": 2 ** 28,
                                                              "norm": "l2",
                                                              "tf": 1.0,
                                                              "idf": None}), procs=8)
wb.dictionary_freeze = True
X_description_train = wb.fit_transform(df_train['description'].fillna(''))
print(X_description_train.shape)
X_description_test = wb.transform(df_test['description'].fillna(''))
print(X_description_test.shape)
print('-')
del(wb)
gc.collect()

num_splits = 5
kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
val_predict = np.zeros(X_train.shape[0])
test_predict = np.zeros(X_test.shape[0])
X_train = X_train.values
X_test = X_test.values
print("X_train shape: ", X_train.shape)
val_scores = []
for train_index, valid_index in kf.split(y):

    X_train_fold, X_valid_fold = X_train[train_index], X_train[valid_index]
    y_train_fold, y_valid_fold = y[train_index], y[valid_index]

    model = CatBoostRegressor(iterations=4000,
                             learning_rate=0.07,
                             depth=10,
                             #loss_function='RMSE',
                             eval_metric='RMSE',
                             random_seed = 23, # reminder of my mortality
                             od_type='Iter',
                             metric_period = 100,
                             od_wait=100,
                             nan_mode="Min",
                             calc_feature_importance=False,
                             l2_leaf_reg=0.8)
    model.fit(X_train_fold, y_train_fold,
                 eval_set=(X_valid_fold, y_valid_fold),
                 cat_features=categorical_features_pos,
                 use_best_model=True,
                 verbose=True)

    val_predict[valid_index] = model.predict(X_valid_fold)
    test_predict += model.predict(X_test) / num_splits
    val_scores.append(rmse(y_valid_fold, model.predict(X_valid_fold)))

validation_score = np.mean(val_scores)
val_predicts = pd.DataFrame(data=val_predict, columns=["deal_probability"])
test_predict = pd.DataFrame(data=test_predict, columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/catboost_oof_yuki_val{}_train.csv'.format(validation_score), index=False)
test_predict.to_csv('../output/catboost_oof_yuki_val{}_test.csv'.format(validation_score), index=False)
print("ExtraTree Val Score: {}".format(validation_score))
notify_line("ExtraTree Val Score: {}".format(validation_score))
