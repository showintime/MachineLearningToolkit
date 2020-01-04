# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 11:06:18 2019

@author: ZWH
"""

import numpy as np

from LayerBase import LayerBase

from Tanh import Tanh
from Sequence import Sequence
from Dense import Dense

'''
standard RNN


'''
class RNN(LayerBase):
    def __init__(self, unit_nums, activation = Tanh()):
        self.unit_nums = unit_nums
        self.layer = Sequence([Dense(unit_nums), activation])
    def forward(self, x):
        '''
        x : ndarray ,shape = [batch_size, length, dims]
        '''
        x = np.transpose(x, axes = [1, 0, 2])
        # now x shape = [length, batch_size, dims]
        self.length = x.shape[0]
        
        self.hs = [np.zero(shape = [x.shape[1], self.unit_nums])]
        # self.hs shape = [batch_size, unit_nums]

        for i in range(self.length):
            concat_hx = np.concatenate([x[i], self.hs[-1]], axis = -1)
            self.hs.append(self.layer(concat_hx))
        
        outputs = np.transpose(np.array(self.hs[1:]), axes = [1, 0, 2])
        
        return outputs
        
    
    def __call__(self, x):
        '''
        x : ndarray ,shape = [batch_size, length, dims]
        '''
        self.x = x
        return self.forward(x)
    
    def apply_gradient(self):
        self.layer.apply_gradient()
    def compute_gradient(self, losses):
        for i in range(self.length):
            pass
        return self.layer.compute_gradient(losses)
    def bachward(self, losses):
        return self.compute_gradient(losses)
    
    
    def build(self, input_shape):
        '''
        这里假设输出的是所有时刻的h，以后会补充到所有选项。
        input_shape = [batch_size, length, dims]
        output_shape = [batch_size, length, unit_nums]
        '''
        output_shape = input_shape
        output_shape[-1] = self.unit_nums
        #这里并不保存layer的输出，因为我们要的是它多次计算的维度，而不是一层dense一样
        _ = self.layer.build(output_shape)

        return output_shape
    
    