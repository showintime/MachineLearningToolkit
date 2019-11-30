# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:40:01 2019

@author: ZWH
"""

W=0.32
B=1.78

from LinearRegression import LinearRegression
import numpy as np

def func(x):
    return W*x+B
x=np.random.random(size=(1000,1))
y=func(x)

lr=LinearRegression()





for zi in range(1000):
    lr.train(x,y,valid_x=x,valid_y=y)
    w=lr.dense.w
    b=lr.dense.b