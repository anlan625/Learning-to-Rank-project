import numpy as np
from learning2rank.rank import RankNet

# -------------------------Set1------------------------------
# Combine train and valid data and use model split to validate
n_sample = 473134+71083
train_X = np.zeros((n_sample, 700))
train_y = np.zeros((n_sample, ))
f = open('set1.train.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                train_X[n][fnum] = fval
            elif i == 0:
                train_y[n] = int(c)
        n += 1
    else:
        break
f.close()
f = open('set1.valid.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                train_X[n][fnum] = fval
            elif i == 0:
                train_y[n] = int(c)
        n += 1
    else:
        break
f.close()

# Test data
n_sample = 165660
test_X = np.zeros((n_sample, 700))
test_y = np.zeros((n_sample, ))
f = open('set1.test.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                test_X[n][fnum] = fval
            elif i == 0:
                test_y[n] = int(c)
        n += 1
    else:
        break
f.close()

# Training
Model = RankNet.RankNet()
Model.fit(train_X, train_y)
pred = Model.predict(test_X)
np.savetxt('RankNet-prediction1.txt', pred, delimiter=',')
print(Model.ndcg(test_y, pred))


# -------------------------Set2------------------------------
# Combine train and valid data and use model split to validate
n_sample = 34815+34881
train_X = np.zeros((n_sample, 700))
train_y = np.zeros((n_sample, ))
f = open('set2.train.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                train_X[n][fnum] = fval
            elif i == 0:
                train_y[n] = int(c)
        n += 1
    else:
        break
f.close()
f = open('set2.valid.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                train_X[n][fnum] = fval
            elif i == 0:
                train_y[n] = int(c)
        n += 1
    else:
        break
f.close()

# Test data
n_sample = 103174
test_X = np.zeros((n_sample, 700))
test_y = np.zeros((n_sample, ))
f = open('set2.test.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        query = line.strip('\n').split(' ')
        for i, c in enumerate(query):
            if i > 1:
                feature = c.split(':')
                fnum = int(feature[0])-1
                fval = float(feature[1])
                test_X[n][fnum] = fval
            elif i == 0:
                test_y[n] = int(c)
        n += 1
    else:
        break
f.close()

# Training set1
Model = RankNet.RankNet()
Model.fit(train_X, train_y)
pred = Model.predict(test_X)
np.savetxt('./RankNet/RankNet-prediction1.txt', pred, delimiter=',')
print(Model.ndcg(test_y, pred))

# Training set2
Model = RankNet.RankNet()
Model.fit(train_X, train_y)
pred = Model.predict(test_X)
np.savetxt('./RankNet/RankNet-prediction2.txt', pred, delimiter=',')
print(Model.ndcg(test_y, pred))
