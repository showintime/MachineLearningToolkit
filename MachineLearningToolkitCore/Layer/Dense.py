# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:16:56 2019

@author: ZWH
"""


import numpy as np
from LayerBase import LayerBase

'''
Dense layer is a trainable layer, which means that it has parameters to train.
Attention to use this layer if you want to reuse this layer parameter.

parameter layer
x shape = [batch_size, input_size]
w shape = [input_size, output_size]
b shape = [output_size]
x @ w + b shape = [batch_size, output_size]

'''
class Dense(LayerBase):
    def __init__(self, unit_nums):
        
        self.weight_shape = [None,unit_nums]
        self.bias_shape = [unit_nums]
    
    def build(self, input_shape):
        '''
        自动推断维度，确定变量维度，并在内存中初始化
        '''
        self.weight_shape[0] = input_shape[-1]
        
        self.w = np.random.random(size = self.weight_shape) * 6 / sum(self.weight_shape)
        self.b = np.zeros(shape = self.bias_shape) + 0.1
        
        self.dw = np.zeros(shape = self.weight_shape)
        self.db = np.zeros(shape = self.bias_shape)
        #print('you only look once')
        output_shape = input_shape
        output_shape[-1] = self.weight_shape[-1]
        return output_shape
    
    def forward(self, x):

        return x @ self.w + self.b
    
    def apply_gradient(self):
        
        self.w -= self.dw
        self.b -= self.db
        
        self.dw *= 0
        self.db *= 0
        
    def compute_gradient(self, losses):
        self.dw += self.x.T @ losses
        self.db += np.sum(losses, axis = 0)
        
        self.dx = losses @ self.w.T
        return self.dx
    def backward(self, losses):
        
        return self.compute_gradient(losses)
    
    def __call__(self,x):
        self.x = x
        return self.forward(x)
    

