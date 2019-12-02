# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:49:59 2019

@author: ZWH
"""
import numpy as np

EPSON=1e-8

class Tanh(object):
    def __init__(self):
        pass
    def tanh(self,x):
        ex=np.exp(x)
        inv_ex=1/(EPSON+ex)
        return (ex-inv_ex)/(ex+inv_ex+EPSON)
    def forward(self,x):
        return self.tanh(x)
    def __call__(self,x):
        self.x=x
        return self.forward(x)
    def backward(self,losses):
        ex=np.exp(self.x)
        inv_ex=1/(EPSON+ex)
        tem=1-((ex-inv_ex)/(ex+inv_ex+EPSON))**2
        return losses*tem