# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:07:23 2021

@author: Anshul
"""

import numpy as np

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.00

def forward(x):
    return w*x

def loss(y,y_predicted):
    return ((y - y_predicted)**2).mean()

# MSE = 1/N(y-y_pred)**2
# dJ/dw = 1/N*(2)(w*x - y)*x
def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Model prediction before  training: {forward(5):.3f}')

learning_rate = 0.01

for epoch in range(20):
    # forward pass
    y_pred = forward(X)
    
    #loss 
    l = loss(Y,y_pred)
    
    # Gradient computation
    dw = gradient(X,Y,y_pred)
    
    w = w - learning_rate*dw
    
    if epoch%1==0:
        print(f'Epoch {epoch+1},  w {w:.3f},   loss {l:.5f}')
        
        

print(f'Model after training : {forward(5):.5f}')