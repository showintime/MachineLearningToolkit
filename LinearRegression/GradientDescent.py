# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 01:24:53 2019

@author: ZWH
"""

class GradientDescent(object):
    def __init__(self,learning_rate=0.1):
        self.learning_rate=learning_rate
    def backward(self,loss):
        loss.backward(self.learning_rate)
    def __call__(self,loss):
        self.backward(loss)