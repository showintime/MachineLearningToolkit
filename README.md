# MachineLearningToolkit[MLTK]

[TOC]

## LinearRegression

### Example

``` python
lr = LinearRegression()

EPOCHES=10
template='Epoch:{},Train_loss:{},Valid_loss:{}'
for epoch in range(EPOCHES):
    train_loss,valid_loss=lr.train(train_X,train_y,valid_x=valid_x,valid_y=valid_y)
    print(template.format(epoch+1,train_loss,valid_loss))
    
test_predictions=lr.predict(test_x)

```

