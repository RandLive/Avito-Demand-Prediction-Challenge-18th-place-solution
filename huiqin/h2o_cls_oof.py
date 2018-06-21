import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML
import re
# Set it according to kernel limits
h2o.init(max_mem_size = "20G",nthreads=20,)

print("\nData Load Stage")
#,nrows=10000
training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
traindex = training.index
testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])#.sample(1000)
testdex = testing.index
y = training.deal_probability.copy()
# training.drop("deal_probability", axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training, testing], axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"] + 0.001)
df["price"].fillna(-999, inplace=True)
df["image_top_1"].fillna(-999, inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day

# Remove Dead Variables
df.drop(["activation_date", "image"], axis=1, inplace=True)

print("\nEncode Variables")
categorical = ["user_id", "region", "city", "parent_category_name", "category_name", "item_seq_number", "user_type",
               "image_top_1"]
# messy_categorical = ["param_1", "param_2", "param_3", "title", "description"]  # Need to find better technique for these
def text_preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"(\\u[0-9A-Fa-f]+)",r"", text)
    text = re.sub(r"===",r" ", text)
    # https://www.kaggle.com/demery/lightgbm-with-ridge-feature/code
    text = " ".join(map(str.strip, re.split('(\d+)',text)))
    regex = re.compile(u'[^[:alpha:]]')
    text = regex.sub(" ", text)
    text = " ".join(text.split())
    return text
df["title"]=df["title"].apply(text_preprocessing)
df["description"]=df["description"].apply(text_preprocessing)
messy_categorical = ["param_1", "param_2", "param_3", "title","description"]
print("Encoding :", categorical + messy_categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
df.drop("title",inplace=True,axis=1)
df.drop("description",inplace=True,axis=1)

X = df.loc[traindex, :].copy()
x1 =list(X.columns)
print("x1=",x1)


# y1='deal_probability'
y1=2#数组里面第3个列
# x1.remove(y1)
print("Training Set shape", X.shape)
test = df.loc[testdex, :].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()

test.drop("deal_probability",axis=1, inplace=True)
from sklearn.model_selection import KFold
modelstart = time.time()
nfold=5
kf = KFold(n_splits=nfold, random_state=42, shuffle=True)
val_predict= np.zeros(y.shape)
aver_rmse=0.0
fold_id = -1
X =X.as_matrix()
print(X[0])
test=test.as_matrix()
print(test[0])
for train_index, val_index in kf.split(X):
    print(len(train_index),len(val_index))
    fold_id += 1
    x_train, x_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    #x_train, x_val = X.loc[train_index], X.loc[val_index]
    #y_train, y_val = y[train_index], y[val_index]


    htrain = h2o.H2OFrame(x_train)
    hval = h2o.H2OFrame(x_val)
    htest = h2o.H2OFrame(test)

    del x_train,x_val
    gc.collect()



    # Set maximum runtime according to Kaggle limits
    # aml = H2OAutoML(max_runtime_secs = 20000)#18000
    aml = H2OAutoML(max_runtime_secs = 20000)
    # aml.train(x=x1,y=y1, training_frame=htrain, validation_frame=hval, leaderboard_frame = hval)
    aml.train( y=y1,training_frame=htrain, validation_frame=hval, leaderboard_frame=hval)

    print('Generate predictions...')
    htrain.drop([y1])
    preds = aml.leader.predict(hval)
    preds = preds.as_data_frame()
    print(preds.shape)
    # train_pred = aml.leader.predict(htrain)

    val_predict[val_index] = np.squeeze(preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    print('RMSLE H2O automl leader: ', rmse)
    aver_rmse += rmse


    test_preds = aml.leader.predict(htest)
    test_preds = test_preds.as_data_frame()
    sub = pd.read_csv('../input/sample_submission.csv')  # , nrows=10000*5

    sub['deal_probability'] = test_preds
    sub['deal_probability'].clip(0.0, 1.0, inplace=True)
    print("Output Prediction CSV")
    sub.to_csv('subm/h2o_submissionV1_{}.csv'.format(fold_id), index=False)
    del preds,test_preds,sub,htrain,hval,htest
    gc.collect()

print("average rmse:{}".format(aver_rmse/nfold))
train_data = pd.read_csv('../input/train.csv')
label = ['deal_probability']
train_user_ids = train_data.user_id.values
train_item_ids = train_data.item_id.values

train_item_ids = train_item_ids.reshape(len(train_item_ids), 1)
train_user_ids = train_item_ids.reshape(len(train_user_ids), 1)
val_predicts = pd.DataFrame(data=val_predict, columns= label)
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
val_predicts.to_csv('subm/h2o_submissionV1_train.csv', index=False)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
#print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

'''
RMSLE H2O automl leader:  0.249451978897

RMSLE H2O automl leader:  0.248917120186
stackedensemble prediction progress: |████████████████████████████████████| 100%
Notebook Runtime: 3.25 Minutes
H2O session _sid_9a3a closed.

x1= ['category_name', 'city', 'deal_probability', 'image_top_1', 'item_seq_number', 'param_1', 'param_2', 'param_3', 'parent_category_name', 'price', 'region', 'user_id', 'user_type', 'Weekday', 'Weekd of Year', 'Day of Month']
Training Set shape (10000, 16)
'''