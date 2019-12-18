# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:50:51 2019

@author: ZWH
"""



class Relu(object):
    def __init__(self):
        pass
    def relu(self, x):
        return self.positive_x_index * x
    def forward(self, x):
        self.positive_x_index = (x >= 0) * 1.0
        return self.relu(x)
    def __call__(self, x):
        self.x = x
        return self.forward(x)
    def backward(self, losses):
        return losses * self.positive_x_index