# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:11:16 2019

@author: ZWH
"""

class Sequence(object):
    def __init__(self,sequencelist):
        self.layerlist=sequencelist
    def add(self,layer):
        self.layerlist.append(layer)
    def forward(self,x):
        for layer in self.layerlist:
            x=layer(x)
        return x
    def backward(self,losses):
        self.layerlist.reverse()
        for layer in self.layerlist:
            losses=layer.backward(losses)
        self.layerlist.reverse()
    def __call__(self,x):
        return self.forward(x)
