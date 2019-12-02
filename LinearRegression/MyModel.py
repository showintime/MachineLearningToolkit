# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:40:01 2019

@author: ZWH
"""

'''
#LinearRegression


import numpy as np
W=np.array([0.54,0.76,0.534,0.675]).reshape(4,1)
B=1.78
def func(x):
    return x@W+B+np.random.random(size=(x.shape[0],1))*0.1
x=np.random.random(size=(1000,4))
y=func(x)

from LinearRegression import LinearRegression

lr=LinearRegression()

TRAIN_NUM=1000
EPOCHES=100
BATCH_SIZE=10
template='Epoch:{:>4}, Train_loss:{:.6}'
for epoch in range(EPOCHES):
    for l in range(0,TRAIN_NUM,BATCH_SIZE):
        
        
        r=min(l+BATCH_SIZE,TRAIN_NUM)
        train_loss=lr.train(x[l:r],y[l:r])
        
    train_loss=lr.loss(lr.predict(x),y)
    print(template.format(epoch+1,train_loss))
   
    
'''





#分类



from mnist_network import smallnetwork


import matplotlib.pyplot as plt

data_path='C:/Users/ZWH/Desktop/MyNeuralNetwork/ANN/pkldata/'
import os
import numpy as np
print(os.listdir(data_path))
def readdata(path):
    import os
    import pickle
    def readadata(path):
        with open(path,'rb') as f:
            data=pickle.load(f)
        return data
    files=os.listdir(path)
    for file in files:
        if 'train' in file:
            traindata=readadata(path+file)
        elif 'valid' in file:
            validdata=readadata(path+file)
        elif 'test' in file:
            testdata=readadata(path+file)
    return traindata,validdata,testdata
(trainx,trainy),(validx,validy),(testx,testy)=readdata(data_path)




k=np.identity(10)
trainy=k[trainy]
validy=k[validy]
testy=k[testy]

sn=smallnetwork()

def show(r=4,c=4):
    plt.figure(figsize=(10,10))
    for i in range(r*c):
        plt.subplot(r,c,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        index=np.random.randint(0,len(trainy))
        pre=np.argmax(sn.predict(trainx[index].reshape(1,-1)))
        lab=np.argmax(trainy[index])
        plt.imshow(trainx[index].reshape(28,-1))
        plt.xlabel('pre:{},lab:{}'.format(pre,lab))
    plt.show()
    


#show()

def acc(x,y):
    predictions=sn.predict(x)
    labels=y
    pre=np.argmax(predictions,axis=1)
    lab=np.argmax(labels,axis=1)
    tem=np.mean((pre==lab)*1.0)

    return tem


atrain_acc=acc(trainx,trainy)
avalid_acc=acc(validx,validy)
atest_acc=acc(testx,testy)

print('train:{},valid:{},test:{}'.format(atrain_acc,avalid_acc,atest_acc))




TRAIN_NUM=len(trainy)
EPOCHES=5
BATCH_SIZE=16
template='Epoch:{:>4}, Train_loss:{:.6}'


for epoch in range(EPOCHES):
    for l in range(0,TRAIN_NUM,BATCH_SIZE):
        
        
        r=min(l+BATCH_SIZE,TRAIN_NUM)
        train_loss=sn.train(trainx[l:r],trainy[l:r])
        
        
    #train_loss=sn.loss(sn.predict(trainx),y)
    print(template.format(epoch+1,train_loss[0]))
#show()

btrain_acc=acc(trainx,trainy)
bvalid_acc=acc(validx,validy)
btest_acc=acc(testx,testy)
print('train:{},valid:{},test:{}'.format(btrain_acc,bvalid_acc,btest_acc))









