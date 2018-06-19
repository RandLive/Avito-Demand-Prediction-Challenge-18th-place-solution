import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from utils import *
import argparse
from fastFM import als

# specify the version.
parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', default='ensemble', help='version')
args = parser.parse_args()
version = args.version
is_read_data = True

train = pd.read_csv('../input/train.csv', usecols=['user_id', 'item_id'])
test = pd.read_csv('../input/test.csv', usecols=['user_id', 'item_id'])
train_user_ids = train.user_id.tolist()
train_item_ids = train.item_id.tolist()
test_user_ids = test.user_id.tolist()
test_item_ids = test.item_id.tolist()
del train, test; gc.collect()

if version=='ensemble':
    if is_read_data:
        X_train, X_test, y = read_train_test_data()
    else:
        X_train = read_parquet("../tmp/X_train.parquet")
        X_test = read_parquet("../tmp/X_test.parquet")
        y = read_parquet("../tmp/y_train.parquet").values.ravel()
    X_tr_sta, X_te_sta, _ = read_train_test_data_stacking()
    X_train = pd.concat([X_train, X_tr_sta],axis=1)
    X_test = pd.concat([X_test, X_te_sta],axis=1)
    del X_tr_sta, X_te_sta; gc.collect()
    nogain_features =[]
    f =open("../tmp/no_gain_features_stack_version.txt")
    for l in f.readlines():
        nogain_features.append(l.replace("\n",""))
    f.close()
    drop_cols = [col for col in X_train.columns if col in nogain_features]
    X = X_train.drop(drop_cols, axis=1)
    X_test = X_test.drop(drop_cols, axis=1)


elif version=='single':
    X = pq.read_table("../tmp/X_train_scaled.parquet").to_pandas().values
    X_test = pq.read_table("../tmp/X_test_scaled.parquet").to_pandas().values
    y = read_parquet("../tmp/y_train.parquet").values.ravel()
    gc.collect()
    print("Done reading data.")


cnt = 0
oof_data_out = {
	"y_val_pred":np.zeros((X.shape[0], 1)),
	"y_test_pred":np.zeros((X_test.shape[0], 1))
}

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_true-y_pred)))


num_splits = 5
kfolds = list(KFold(n_splits=num_splits, random_state=42, shuffle=True).split(X))

for ix_train, ix_valid in kfolds:
	print("============ROUND{}==============".format(cnt+1))
	cnt+=1
	X_train = X[ix_train,:]
	X_val = X[ix_valid, :]
	y_train = y[ix_train]
	y_val = y[ix_valid]
    model = als.FMRegression(
        n_iter=2000,
        init_stdev=0.1,
        rank=30,
        l2_reg_w=50,
        l2_reg_V=30000
        )
    model.fit(X_train, y_train)

    val_predict[valid_index] = model.predict(X_valid_fold)
    test_predict += model.predict(X_test) / num_splits
    val_scores.append(rmse(y_valid_fold, model.predict(X_valid_fold)))

validation_score = np.mean(val_scores)
val_predicts = pd.DataFrame(data=oof_data_out["y_val_pred"], columns=["deal_probability"])
test_predict = pd.DataFrame(data=oof_data_out["y_test_pred"], columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/fm_oof_yuki_val{}_train.csv'.format(validation_score), index=False)
test_predict.to_csv('../output/fm_oof_yuki_val{}_test.csv'.format(validation_score), index=False)
