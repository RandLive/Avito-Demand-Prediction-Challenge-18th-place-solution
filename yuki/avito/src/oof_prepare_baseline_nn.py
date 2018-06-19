import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from utils import *
import argparse

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
    X_train = X_train.drop(drop_cols, axis=1)
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

# if loss_function == "root_mean_squared_error":
# 	loss = root_mean_squared_error
# else:
# 	loss = loss_function

num_splits = 5
kfolds = list(KFold(n_splits=num_splits, random_state=42, shuffle=True).split(X))

for ix_train, ix_valid in kfolds:
	print("============ROUND{}==============".format(cnt+1))
	cnt+=1
	X_train = X[ix_train,:]
	X_val = X[ix_valid, :]
	y_train = y[ix_train]
	y_val = y[ix_valid]

	X_train, X_train_val, y_train, y_train_val = train_test_split(X_train, y_train, test_size=.15, random_state=1234)

	# modeling
	print("training")
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.layers.normalization import BatchNormalization
	from keras.optimizers import SGD
	from keras.callbacks import EarlyStopping, ModelCheckpoint
	from keras.layers.advanced_activations import LeakyReLU
	from keras.callbacks import Callback
	from keras import backend as K


	params = {
	    "M1":1000,
	    "M2":2000,
	    "M3":3000,
	    "M4":2000,
	    "M5":1000,
	    "learning_rate":0.04,
	    "leaky_rate":0.2,
	    "batch_size":128,
	    "drop_rate":0.2,
	    "epochs":4000
	}

	N = X_train.shape[0]
	D = X_train.shape[1]
	M1 = params["M1"]
	M2 = params["M2"]
	M3 = params["M3"]
	M4 = params["M4"]
	M5 = params["M5"]
	leaky_rate = params["leaky_rate"]

	model = Sequential()
	model.add(Dense(M1, input_shape=(D,), activation="linear"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=leaky_rate))
	model.add(Dropout(params["drop_rate"]))
	model.add(Dense(M2, activation="linear"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=leaky_rate))
	model.add(Dropout(params["drop_rate"]))
	model.add(Dense(M3, activation="linear"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=leaky_rate))
	model.add(Dropout(params["drop_rate"]))
	model.add(Dense(M4, activation="linear"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=leaky_rate))
	model.add(Dropout(params["drop_rate"]))
	model.add(Dense(M5, activation="linear"))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=leaky_rate))
	model.add(Dropout(0.1))
	model.add(Dense(1))
	model.add(Activation("sigmoid"))


	print("======================================================")
	print("start adam optimizer...")
	print("======================================================")
	batch_size = params["batch_size"]
	nb_epoch = params["epochs"]
	model.compile(loss=root_mean_squared_error, optimizer="adam", metrics=[root_mean_squared_error])
	print(model.summary())

	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	bst_model_path = 'dnn_oof.h5'
	model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=20, verbose=1
	            , shuffle=True, validation_data=(X_train_val, y_train_val)
	            , callbacks=[early_stopping, model_checkpoint])
	model.load_weights(bst_model_path)
	print("======================================================")
	print("start sgd optimizer...")
	print("======================================================")
	sgd = SGD(lr=params["learning_rate"], momentum=0.9, decay=0.0, nesterov=True)
	model.compile(loss=root_mean_squared_error, optimizer=sgd, metrics=[root_mean_squared_error])
	early_stopping =EarlyStopping(monitor='val_loss', patience=5)
	model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=20, verbose=1
	            , shuffle=True, validation_data=(X_train_val, y_train_val)
	            , callbacks=[early_stopping, model_checkpoint])

	model.load_weights(bst_model_path)

	oof_data_out["y_test_pred"] += model.predict(X_test, batch_size=256) / len(kfolds)
	oof_data_out["y_val_pred"][ix_valid, :] = model.predict(X_val, batch_size=256)

	del model; gc.collect()
	del X_train, X_val, y_train, y_val; gc.collect()
	K.clear_session()

val_predicts = pd.DataFrame(data=oof_data_out["y_val_pred"], columns=["deal_probability"])
test_predict = pd.DataFrame(data=oof_data_out["y_test_pred"], columns=["deal_probability"])
val_predicts['user_id'] = train_user_ids
val_predicts['item_id'] = train_item_ids
test_predict['user_id'] = test_user_ids
test_predict['item_id'] = test_item_ids
val_predicts.to_csv('../output/mlp_oof_yuki_train.csv', index=False)
test_predict.to_csv('../output/mlp_oof_yuki_test.csv', index=False)
