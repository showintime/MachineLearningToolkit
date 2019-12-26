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
        self.h_shape = [unit_nums]
        self.layer = Sequence([Dense(unit_nums), activation])
    def forward(self, x):
        '''
        x : ndarray ,shape = [batch_size, length, dims]
        '''
        
        x = np.transpose(x, axes = [1, 0, 2])
        # now x shape = [length, batch_size, dims]
        length = x.shape[0]
        
        self.hs = [np.zero(shape = [x.shape[1], self.h_shape])]
        # self.h shape = [batch_size, unit_nums]
        
        
        
        for i in range(lenght):
            concat = np.concatenate([x[i], self.hs[-1]], axis = -1)
            self.hs.append(self.layer(concat))
        
        outputs = np.transpose(np.array(self.hs[1:]), axes = [1, 0, 2])
        
        return outputs
        
    
    def __call__(self, x):
        '''
        x : ndarray ,shape = [batch_size, length, dims]
        '''
        self.x = x
        return self.forward(x)
    
    
    def bachward(self):
        pass
    
    
    def build(self, input_shape):
        self.h = np.zeros(self.h_shape)
        
        output_shape = [None, self.h_shape + input_shape[-1]]
        output_shape = self.layer.build(output_shape)
    
    