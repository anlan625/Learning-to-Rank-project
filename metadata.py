import numpy as np
from learning2rank.rank import RankNet

# -------------------------Produce training data------------------------------
# Combine train and valid data and use model split to validate
n_sample = 473134+71083+34815+34881
train_X = np.zeros((n_sample, 700))
train_y = np.zeros((n_sample, ))
f = open('./data/set1.train.txt', "r")
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
f = open('./data/set1.valid.txt', "r")
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
f = open('./data/set2.train.txt', "r")
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
f = open('./data/set2.valid.txt', "r")
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
n_sample = 165660 + 103174
test_X = np.zeros((n_sample, 700))
test_y = np.zeros((n_sample, ))
f = open('./data/set1.test.txt', "r")
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
f = open('./data/set2.test.txt', "r")
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

# -------------------------Partition for stack model-----------------------------
partition = int(len(train_X)/3)
subset1_x = train_X[0: partition]
subset2_x = train_X[partition: 2*partition]
subset3_x = train_X[2*partition: 3*partition]
subset1_y = train_y[0: partition]
subset2_y = train_y[partition: 2*partition]
subset3_y = train_y[2*partition: 3*partition]

model1_x = np.vstack([subset1_x, subset2_x])
model2_x = np.vstack([subset1_x, subset3_x])
model3_x = np.vstack([subset2_x, subset3_x])

s1 = list(subset1_y)
s2 = list(subset2_y)
s3 = list(subset3_y)

model1_y = []
model2_y = []
model3_y = []

model1_y.extend(s1)
model1_y.extend(s2)

model2_y.extend(s1)
model2_y.extend(s3)

model3_y.extend(s2)
model3_y.extend(s3)

model1_y = np.array(model1_y)
model2_y = np.array(model2_y)
model3_y = np.array(model3_y)

# -------------------------Generate Meta data----------------------------
Model = RankNet.RankNet()
Model.fit(model1_x, model1_y)
train1 = Model.predict(subset3_x)
test1 = Model.predict(test_X)

Model.fit(model2_x, model2_y)
train2 = Model.predict(subset2_x)
test2 = Model.predict(test_X)

Model.fit(model3_x, model3_y)
train3 = Model.predict(subset1_x)
test3 = Model.predict(test_X)

train_x = np.vstack([train1,train2,train3])
train_y = []
train_y.extend(s3)
train_y.extend(s2)
train_y.extend(s1)

test = (test1 + test2 + test3)/3
np.savetxt('./data/rn_meta_test_x.txt', test, delimiter=',')
np.savetxt('./data/rn_meta_test_y.txt', test_y, delimiter=',')
np.savetxt('./data/rn_meta_train_x.txt', train_x, delimiter=',')
np.savetxt('./data/rn_meta_train_y.txt', train_y, delimiter=',')
