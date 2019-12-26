# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:11:16 2019

@author: ZWH
"""

from LayerBase import LayerBase

class Sequence(LayerBase):
    def __init__(self, sequencelist):
        self.layerlist = sequencelist
    def add(self, layer):
        self.layerlist.append(layer)
    
    def apply_gradient(self):
        for layer in self.layerlist:
            layer.apply_gradient()
    
    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x
    def compute_gradient(self, losses):
        self.layerlist.reverse()
        
        for layer in self.layerlist:
            losses = layer.backward(losses)

        self.layerlist.reverse()
        return losses
    def backward(self, losses):
        return self.compute_gradient(losses)
    
    def __call__(self, x):
        return self.forward(x)
    
    def build(self, input_shape):
        output_shape = input_shape
        for layer in self.layerlist:
            output_shape = layer.build(output_shape)
        return output_shape

