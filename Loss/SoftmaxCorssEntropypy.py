# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:09:18 2019

@author: ZWH
"""

import numpy as np

EPSON=1e-8

class SoftmaxCrossEntropy(object):
    def __init__(self):
        pass
    def backward(self,learning_rate):
        loss=learning_rate/self.batch_size*(-1/(self.predictions+EPSON)*self.labels)
        return loss
    def forward(self,predictions,labels):
        self.batch_size=self.predictions.shape[0]
        self.batch_loss=-self.labels*np.log(self.predictions)
        return np.mean(np.sum(self.batch_loss,axis=-1))
    def __call__(self,predictions,labels):
        self.predictions=predictions
        self.labels=labels
        return self.forward(predictions,labels)