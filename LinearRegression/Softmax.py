# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:42:53 2019

@author: ZWH
"""

import numpy as np
class Softmax(object):
    def __init__(self):
        pass
    def softmax(self,x):
        '''
        在最后一个维度做softmax变换
        '''
        x=x-np.max(x,axis=-1,keepdims=True)
        ex=np.exp(x)
        x=ex/np.sum(ex,axis=-1,keepdims=True)
        return x
    def forward(self,x):
        return self.softmax(x)
    def __call__(self,x):
        self.x=x
        return self.forward(x)
    def backward(self,losses):
        x=self.x-np.max(self.x,axis=-1,keepdims=True)
        ex=np.exp(x)
        x=ex/np.sum(ex,axis=-1,keepdims=True)
        #now x is softmax
        tem00=np.expand_dims(x,axis=-1)
        tem01=np.expand_dims(x,axis=-2)
        tem=tem00@tem01
        tem1=np.expand_dims(losses,axis=-2)@tem
        tem11=np.sum(tem1,axis=-2)
        tem2=losses*x
        tem3=tem2-tem11
        return tem3

