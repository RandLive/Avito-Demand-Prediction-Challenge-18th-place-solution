import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import argparse
from scipy.special import erfinv
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--loss', '-l', default="root_mean_squared_error", help='loss function')
parser.add_argument('--arch', '-a', default="base", help='archtechture')
args = parser.parse_args()
# root_mean_squared_error, binary_crossentropy, kullback_leibler_divergence
loss_function = args.loss
arch = args.arch
print("====================Start====================")
print("Loss Function: {}".format(loss_function))
print("Architecture: {}".format(arch))
class GaussRankScaler():

	def __init__( self ):
		self.epsilon = 0.001
		self.lower = -1 + self.epsilon
		self.upper =  1 - self.epsilon
		self.range = self.upper - self.lower

	def fit_transform( self, X ):

		i = np.argsort( X, axis = 0 )
		j = np.argsort( i, axis = 0 )

		assert ( j.min() == 0 ).all()
		assert ( j.max() == len( j ) - 1 ).all()

		j_range = len( j ) - 1
		self.divider = j_range / self.range

		transformed = j / self.divider
		transformed = transformed - self.upper
		transformed = erfinv( transformed )

		return transformed


is_read_data = False
if is_read_data:
	from utils import read_train_test_data#,tmp_read_train_valid
	X, X_test, _ = read_train_test_data_all()
	# drop label encoding
	lbl_cols = [col for col in X.columns if "_labelencod" in col or "oof_" in col]
	price_cols = [col for col in X.columns if "price" in col]
	X.drop(lbl_cols, axis=1, inplace=True)
	X_test.drop(lbl_cols, axis=1, inplace=True)


	# nan index
	train_nan_idx = csr_matrix((np.isnan(X)).astype(int))
	test_nan_idx = csr_matrix((np.isnan(X_test)).astype(int))
	X = X.fillna(X.median())#X.fillna(X.median()) # X.fillna(0)
	X = X.replace(np.inf, 9999.999)
	X = X.replace(-np.inf, -9999.999)
	X = X.values
	X_test = X_test.fillna(X_test.median())#X_test.fillna(X_test.median())
	X_test = X_test.replace(np.inf, 99999999.999)
	X_test = X_test.replace(-np.inf, -99999999.999)
	X_test = X_test.values
	train_size = X.shape[0]
	# X = X.fillna(0)
	# X = X.replace(np.inf, 9999.999)
	# X = X.replace(-np.inf, -9999.999)
	# X = X.values
	# X_test = X_test.fillna(0)
	# X_test = X_test.replace(np.inf, 9999.999)
	# X_test = X_test.replace(-np.inf, -9999.999)
	# X_test = X_test.values
	# train_size = X.shape[0]
	# gc.collect()

	print("scale data")
	scaler = GaussRankScaler()#StandardScaler() # GaussRankScaler()
	X_all = scaler.fit_transform(np.r_[X, X_test])
	del X, X_test; gc.collect()
	# X = pd.DataFrame(X_all[:train_size,:])
	X = pd.DataFrame(X_all[:train_size,:] * np.array((train_nan_idx.todense()==0).astype(int)))
	del train_nan_idx
	print("Done scaling train data...")
	# X_test = pd.DataFrame(X_all[train_size:,:])
	X_test = pd.DataFrame(X_all[train_size:,:] * np.array((test_nan_idx.todense()==0).astype(int)))
	print("Done scaling test data...")
	# del X_all; gc.collect()
	del X_all, test_nan_idx;gc.collect()
	arrow_table_train = pa.Table.from_pandas(X)
	arrow_table_test = pa.Table.from_pandas(X_test)
	pq.write_table(arrow_table_train, "../tmp/X_train_nooof_scaled.parquet")
	pq.write_table(arrow_table_test, "../tmp/X_test_nooof_scaled.parquet")
else:
	print("read parquet data...")
	X = pq.read_table("../tmp/X_train_nooof_scaled.parquet").to_pandas()
	X_test = pq.read_table("../tmp/X_test_nooof_scaled.parquet").to_pandas()

X_test = X_test.values
X = X.values
y = pd.read_csv("../input/train.csv")["deal_probability"].values
nsplits = 5
# kfolds = KFold(n_splits=nsplits)
with open("../tmp/oof_index.dat", "rb") as f:
	kfolds = dill.load(f)

val_score = []
result = np.zeros((X_test.shape[0], 1))
cnt = 0

oof_data_out = {
	"y_val_pred":np.zeros((X.shape[0], 1)),
	"y_test_pred":np.zeros((X_test.shape[0], 1))
}

def root_mean_squared_error(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_true-y_pred)))

if loss_function == "root_mean_squared_error":
	loss = root_mean_squared_error
else:
	loss = loss_function

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


	if arch == "base":
		params = {
		    "M1":3000,
		    "M2":4500,
		    "M3":5500,
		    "M4":4500,
		    "M5":3000,
		    "learning_rate":0.04,
		    "leaky_rate":0.2,
		    "batch_size":64,
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
	elif arch == "bottleneck":
		params = {
		    "M1":3000,
		    "M2":4500,
		    "M3":3000,
		    "M4":4500,
		    "M5":3000,
		    "learning_rate":0.04,
		    "leaky_rate":0.2,
		    "batch_size":64,
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
	model.compile(loss=loss, optimizer="adam", metrics=[root_mean_squared_error])
	print(model.summary())

	early_stopping =EarlyStopping(monitor='val_loss', patience=3)
	bst_model_path = 'dnn_{}_{}.h5'.format(arch,loss_function)
	model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1
	            , shuffle=True, validation_data=(X_train_val, y_train_val)
	            , callbacks=[early_stopping, model_checkpoint])
	model.load_weights(bst_model_path)
	print("======================================================")
	print("start sgd optimizer...")
	print("======================================================")
	sgd = SGD(lr=params["learning_rate"], momentum=0.9, decay=0.0, nesterov=True)
	model.compile(loss=loss, optimizer=sgd, metrics=[root_mean_squared_error])
	early_stopping =EarlyStopping(monitor='val_loss', patience=5)
	model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
	model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1
	            , shuffle=True, validation_data=(X_train_val, y_train_val)
	            , callbacks=[early_stopping, model_checkpoint])

	model.load_weights(bst_model_path)

	oof_data_out["y_test_pred"] += model.predict(X_test, batch_size=256) / len(kfolds)
	oof_data_out["y_val_pred"][ix_valid, :] = model.predict(X_val, batch_size=256)

	del X_train, X_val, y_train, y_val; gc.collect()


output_labels = ["oof_stacking_level1_mlp_nooof_{}_{}".format(arch,loss_function)]
df_oof_test = pd.DataFrame(oof_data_out["y_test_pred"], columns=output_labels)
df_oof_val = pd.DataFrame(oof_data_out["y_val_pred"], columns=output_labels)
to_parquet(df_oof_val,"../features/oof_stacking_level1_mlp_nooof_{}_{}_train.parquet".format(arch,loss_function))
to_parquet(df_oof_test, "../features/oof_stacking_level1_mlp_nooof_{}_{}_test.parquet".format(arch,loss_function))
notify_line("Done Training. CV Score")
