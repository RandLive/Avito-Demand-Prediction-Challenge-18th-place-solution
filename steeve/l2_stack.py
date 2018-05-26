import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import os
import gc

nb_models = 24
with open(f'../input/layer_2_{nb_models}m_train.p', 'rb') as f:
    x_train = pickle.load(f)
    
with open(f'../input/layer_2_{nb_models}m_test.p', 'rb') as f:
    x_test = pickle.load(f)    
    
with open('../input/resnet50_500_train_df.p', 'rb') as f:
    y_train = pickle.load(f).deal_probability.values
    
    
from sklearn.metrics import mean_squared_error

def train_bagging(X, y, fold_count):
    global fname
    parameters = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,

        'verbose': -1
    }
    
    kf = KFold(n_splits=fold_count, random_state=42, shuffle=True)
#     skf = StratifiedKFold(n_splits=fold_count, random_state=None, shuffle=False)
    fold_id = -1
    model_list = []

    rmse_list = []
    for train_index, test_index in kf.split(X):
        
        fold_id +=1 
        
#         if fold_id !=0: continue
        print(f'fold number: {fold_id}', flush=True)
        
        
        x_train, x_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]

        model_path = f'../weights/{fname}_fold{fold_id}.model'
        # if model weights exist
#         if os.path.exists(model_path):
#             model = lgb.Booster(model_file=model_path)
#             y_pred = model.predict(x_val)        

#             model_list.append(model)
#             rmse = mean_squared_error(y_val, y_pred) ** 0.5
#             rmse_list.append(rmse)
#             continue

        tr_data = lgb.Dataset(x_train, label=y_train)
        va_data = lgb.Dataset(x_val, label=y_val, reference=tr_data)
        del x_train, y_train
        gc.collect()
        model = lgb.train(parameters,
                  tr_data,
                  valid_sets=va_data,
                  num_boost_round=5000,
                  early_stopping_rounds=50,
                  verbose_eval=500)
        
        model.save_model(model_path)
        y_pred = model.predict(x_val)        

        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        print(f'rmse: {rmse}', flush=True)
        rmse_list.append(rmse)
        model_list.append(model)
    print(f'rmse score: {np.mean(rmse_list)}', flush=True)
    return model_list






# fname = 'des_word_10000_nwords_title_5000_params_5000_lower_swr_resnet50_500_logprice_params_incep30_lgb_5fold'
fname = f'l2_{nb_models}m_lgb_5fold'
print(f'file name: {fname}', flush=True)

model_list = train_bagging(x_train, y_train, 5)
print(f"model list length: {len(model_list)}")
# fname = 'des_word_10000_nwords_title_5000_lower_swr_resnet50_500_logprice_params_lgb_11old'

sub = pd.read_csv('../input/sample_submission.csv')

for index, model in enumerate(model_list):
    if index == 0: 
        y_pred = model.predict(x_test)
        
    else:
        y_pred *= model.predict(x_test)
 
    
y_pred = np.clip(y_pred, 0, 1)
y_pred = y_pred **( 1.0/ len(model_list))


sub = pd.read_csv('../input/sample_submission.csv')
sub['deal_probability'] = y_pred
sub['deal_probability'].clip(0.0, 1.0, inplace=True)
sub.to_csv(f'../output/{fname}_test.csv', index=False) 

cmd = f'kaggle competitions submit -c avito-demand-prediction -f ../output/{fname}_test.csv -m cv:{np.mean(rmse_list)}'
subprocess.call(cmd.split())