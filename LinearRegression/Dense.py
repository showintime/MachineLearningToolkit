# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 00:16:56 2019

@author: ZWH
"""


import numpy as np


'''
x shape=[batch_size,input_size]
w shape=[input_size,output_size]
b shape=[output_size]
x@w+b shape=[batch_size,output_size]

'''
class Dense(object):
    def __init__(self,unit_nums):

        self.issuccessinit=False
        self.shape=[None,unit_nums]
    def realinit(self,shape):
        '''
        calculte x@self.w+self.b
        '''
        self.w=np.random.random(size=shape)
        self.b=np.random.random(size=(shape[1]))
        #print('you only look once')
        self.issuccessinit=True
    def forward(self,x):
        return x@self.w+self.b
    def applygradient(self,dw,db):
        self.w-=dw
        self.b-=db
    def backward(self,losses):
        dw=self.x.T@losses
        db=np.sum(losses,axis=0)
        self.applygradient(dw,db)
        dx=losses@self.w.T
        return dx
    def __call__(self,x):
        self.x=x
        if not self.issuccessinit:
            self.shape[0]=x.shape[1]
            self.realinit(shape=self.shape)
        return self.forward(x)
    

