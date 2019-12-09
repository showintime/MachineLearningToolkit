# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:40:46 2019

@author: ZWH
"""

import numpy as np

class MeanAbsoluteError(object):
    def __init__(self):
        pass
    def backward(self,learning_rate):
        
        loss=learning_rate/self.batch_size*((self.predictions>self.labels)*1+(self.predictions<self.labels)*(-1))
        return loss
    
    def forward(self,predictions,labels):
        self.batch_size=self.predictions.shape[0]
        self.batch_loss=np.abs(predictions-labels)
        return np.mean(np.sum(self.batch_loss,axis=-1))
    def __call__(self,predictions,labels):
        self.predictions=predictions
        self.labels=labels
        return self.forward(predictions,labels)