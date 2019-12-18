
# Machine Learning Toolkit MLTK

[TOC]

## Linear Regression

### Example

``` python
lr = LinearRegression()

EPOCHES=10
template='Epoch:{},Train_loss:{},Valid_loss:{}'
for EPOCH in range(EPOCHES):
	train_loss,valid_loss=lr.train(train_X,train_y,valid_x=valid_x,valid_y=valid_y)
    print(template.format(EPOCH+1,train_loss,valid_loss))
    
test_predictions=lr.predict(test_x)

```

**Commonly,learning rate<1,or it will not be converge! **

## Classification Regression
### Example
``` python
'''
handwritten digits recognizer
MNIST Dataset
train data number 50000
valid data number 10000
test  data number 10000
'''
from mnist_network import smallnetwork
'''
model=Sequence([Dense(784),Tanh(),Dropout(0.5),Dense(10),Softmax()])
loss=LogisticSoftmaxCrossEntropy()
optimizer=GradientDescent()
'''
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

acc_template='train_acc:{}, valid_acc:{}, test_acc:{}'
print(acc_template.format(acc(trainx,trainy),acc(validx,validy),acc(testx,testy)))



TRAIN_NUM=len(trainy)
EPOCHES=20
BATCH_SIZE=32
template='Epoch:{:>4}, Train_loss:{:.6}'


for epoch in range(EPOCHES):
    for l in range(0,TRAIN_NUM,BATCH_SIZE):
        
        
        r=min(l+BATCH_SIZE,TRAIN_NUM)
        train_loss=sn.train(trainx[l:r],trainy[l:r])
        
    btrain_acc=acc(trainx,trainy)
    bvalid_acc=acc(validx,validy)
    btest_acc=acc(testx,testy)
    train_loss=sn.loss(sn.predict(trainx),trainy)
    print(template.format(epoch+1,train_loss[0]))
    print('train:{},valid:{},test:{}'.format(btrain_acc,bvalid_acc,btest_acc))

```

### Result:

```
train:0.0989,valid:0.098,test:0.0954
Epoch:   1, Train_loss:0.35232
train:0.89194,valid:0.9065,test:0.8976
Epoch:   2, Train_loss:0.275229
train:0.91518,valid:0.9237,test:0.9186
Epoch:   3, Train_loss:0.232661
train:0.93056,valid:0.938,test:0.9283
Epoch:   4, Train_loss:0.205994
train:0.93746,valid:0.9452,test:0.9387
Epoch:   5, Train_loss:0.194031
train:0.94202,valid:0.9447,test:0.9425
Epoch:   6, Train_loss:0.187288
train:0.94284,valid:0.9464,test:0.9396
```