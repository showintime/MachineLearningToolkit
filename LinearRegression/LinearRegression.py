# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:16:56 2019

@author: ZWH
"""


from Dense import Dense
from SquareLoss import SquareLoss
from GradientDescent import GradientDescent


class LinearRegression(object):
    
    def __init__(self):
        self.dense=Dense(1)
        self.loss=SquareLoss()
        self.optimizer=GradientDescent()
        
        
    def predict(self,x):
        x=self.dense(x)
        return x
    def forward(self,x):
        return self.predict(x)
    def backward(self,loss):
        pass
    def train(self,train_x,train_y,valid_x=None,valid_y=None):
		
        train_predictions=self.forward(train_x)
        train_loss=self.loss(train_predictions,train_y)
        losslist=[train_loss]
        if valid_x is not None:
            valid_predictions=self.dense(valid_x)
            valid_loss=self.loss(valid_predictions,valid_y)
            losslist.append(valid_loss)
        losses=self.optimizer(self.loss)
        self.dense.backward(losses)
        return losslist
        
        