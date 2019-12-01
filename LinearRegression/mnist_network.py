# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:00:26 2019

@author: ZWH
"""

from Dense import Dense
from SquareLoss import SquareLoss
from GradientDescent import GradientDescent
from Sequence import Sequence
from Softmax import Softmax

class smallnetwork(object):
    def __init__(self):
        self.model=Sequence([Dense(10),Softmax()])
        self.loss=SquareLoss()
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