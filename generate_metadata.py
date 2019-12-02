#!/usr/bin/env python
# coding: utf-8

# In[4]:


train = []
f = open('./data/set1.train.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        train.append(line)
    else:
        break
f.close()
f = open('./data/set1.valid.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        train.append(line)
    else:
        break
f.close()
f = open('./data/set2.train.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        train.append(line)
    else:
        break
f.close()
f = open('./data/set2.valid.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        train.append(line)
    else:
        break
f.close()


# In[9]:


partition = int(len(train)/3)
subset1 = train[0: partition]
subset2 = train[partition: 2*partition]
subset3 = train[2*partition: 3*partition]

model1 = []
model2 = []
model3 = []

model1.extend(subset1)
model1.extend(subset2)

model2.extend(subset1)
model2.extend(subset3)

model3.extend(subset2)
model3.extend(subset3)


# In[12]:


file = open("./data/subset1.txt","w")
for pair in subset1:
    file.writelines(pair) 
file.close()


# In[13]:


file = open("./data/subset2.txt","w")
for pair in subset2:
    file.writelines(pair) 
file.close()


# In[14]:


file = open("./data/subset3.txt","w")
for pair in subset3:
    file.writelines(pair) 
file.close()


# In[15]:


file = open("./data/model1_train.txt","w")
for pair in model1:
    file.writelines(pair) 
file.close()


# In[16]:


file = open("./data/model2_train.txt","w")
for pair in model2:
    file.writelines(pair) 
file.close()


# In[17]:


file = open("./data/model3_train.txt","w")
for pair in model3:
    file.writelines(pair) 
file.close()


# In[18]:


test = []
f = open('./data/set1.test.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        test.append(line)
    else:
        break
f.close()
f = open('./data/set2.test.txt', "r")
n = 0
while True:
    line = f.readline()
    if line:
        test.append(line)
    else:
        break
f.close()
file = open("./data/test.txt","w")
for pair in test:
    file.writelines(pair) 
file.close()


# In[ ]:




