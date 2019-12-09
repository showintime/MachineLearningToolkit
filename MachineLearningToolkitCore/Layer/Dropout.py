# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:30:48 2019

@author: ZWH
"""

import numpy as np

class Dropout(object):
    def __init__(self,rate=0.5):
        self.rate=rate
    def dropout(self,x):
        return x*self.dropmatrix
    def forward(self,x):
        self.dropmatrix=(np.random.random(size=x.shape)>self.rate)*1.0
        return self.dropout(x)
    def __call__(self,x):
        self.x=x
        return self.forward(x)
    def backward(self,losses):
        return losses*self.dropmatrix