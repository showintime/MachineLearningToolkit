# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:40:01 2019

@author: ZWH
"""

import numpy as np
W=np.array([0.54,0.76,0.534,0.675]).reshape(4,1)
B=1.78

from LinearRegression import LinearRegression
import numpy as np

def func(x):
    return x@W+B#+np.random.random(size=(x.shape[0],1))*0.0001
x=np.random.random(size=(1000,4))
y=func(x)

lr=LinearRegression()




template='{},{}'
for zi in range(10000):
    tl,vl=lr.train(x,y,valid_x=x,valid_y=y)
    print(template.format(tl,vl))
    w=lr.dense.w
    b=lr.dense.b