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
    def backward(self):
        pass
    def forward(self,predictions,labels):
        maxis=tuple(list(range(len(predictions.shape)))[1:])
        self.batch_loss=np.sum(np.square(predictions-labels),axis=maxis)
        return np.mean(self.batch_loss)
    def __call__(self,predictions,labels):
        self.predictions=predictions
        self.labels=labels
        return self.forward(predictions,labels)