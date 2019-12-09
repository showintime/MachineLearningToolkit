# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:16:56 2019

@author: ZWH
"""


from MachineLearningToolkitCore.Layer.Dense import Dense
from MachineLearningToolkitCore.Loss.MeanSquaredError import MeanSquaredError
from MachineLearningToolkitCore.Optimizer.GradientDescent import GradientDescent
from MachineLearningToolkitCore.Layer.Sequence import Sequence

class LinearRegression(object):
    
    def __init__(self):
        self.model=Sequence([Dense(1)])
        self.loss=MeanSquaredError()
        self.optimizer=GradientDescent()
        
        
    def predict(self,x):
        x=self.model(x)
        return x
    def forward(self,x):
        return self.predict(x)
    def __call__(self,x):
        return self.forward(x)
    def backward(self,loss):
        pass
    def train(self,train_x,train_y,valid_x=None,valid_y=None):
		
        train_predictions=self.forward(train_x)
        train_loss=self.loss(train_predictions,train_y)
        losslist=[train_loss]
        if valid_x is not None:
            valid_predictions=self.forward(valid_x)
            valid_loss=self.loss(valid_predictions,valid_y)
            losslist.append(valid_loss)
        losses=self.optimizer(self.loss)
        self.model.backward(losses)
        return losslist
        
