# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 01:21:54 2019

@author: ZWH
"""
import numpy as np

'''
1/(2*batch_size)*Sigma{[y_pred(i)-y_true(i))]^2}
'''
class SquareLoss(object):
    def __init__(self):
        pass
    def backward(self,learning_rate):
        loss=learning_rate/self.predictions.shape[0]*(self.predictions-self.labels)
        return loss
    
    def forward(self,predictions,labels):
        self.batch_loss=np.square(predictions-labels)
        return np.mean(np.sum(self.batch_loss,axis=-1))
    def __call__(self,predictions,labels):
        self.predictions=predictions
        self.labels=labels
        return self.forward(predictions,labels)