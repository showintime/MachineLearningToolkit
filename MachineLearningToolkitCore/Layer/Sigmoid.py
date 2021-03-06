# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 13:04:50 2019

@author: ZWH
"""

import numpy as np
from LayerBase import LayerBase
class Sigmoid(LayerBase):
    def __init__(self):
        pass
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    def forward(self, x):
        return self.sigmoid(x)
    def __call__(self, x):
        self.x = x
        return self.forward(x)
    def backward(self, losses):
        ex=np.exp(-self.x)
        return losses * ex / ((1 + ex) ** 2)

