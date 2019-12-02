import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np

# -------------------------Generate Meta data----------------------------
meta_train_x = []
f = open('./data/rn_meta_train_x.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_train_x.append(float(query))
    else:
        break
f.close()

f = open('./data/rs_meta_train_x.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_train_x.append(float(query))
    else:
        break
f.close()

meta_train_y = []
f = open('./data/rn_meta_train_y.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_train_y.append(float(query))
    else:
        break
f.close()

f = open('./data/rs_meta_train_y.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_train_y.append(float(query))
    else:
        break
f.close()

meta_test_x = []
f = open('./data/rn_meta_test_x.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_test_x.append(float(query))
    else:
        break
f.close()

f = open('./data/rs_meta_test_x.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_test_x.append(float(query))
    else:
        break
f.close()

meta_test_y = []
f = open('./data/rn_meta_test_y.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_test_y.append(float(query))
    else:
        break
f.close()

f = open('./data/rs_meta_test_y.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n')
        meta_test_y.append(float(query))
    else:
        break
f.close()

# multi-class classification with Keras
# load dataset
X = meta_train_x
Y = meta_train_y
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=500, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))