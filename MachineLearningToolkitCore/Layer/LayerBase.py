# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:26:33 2019

@author: ZWH
"""


class LayerBase(object):
    def __init__(self):
        pass
    def forward(self):
        pass
    def backward(self):
        pass
    def build(self, input_shape):
        '''
        确定参数，并在内存中初始化
        '''
        output_shape = input_shape
        return output_shape
    
    def compute_gradient(self, loss):
        pass
    def apply_gradient(self):
        pass
        
        