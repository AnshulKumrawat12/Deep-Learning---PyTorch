# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:26:50 2021

@author: Anshul
"""

import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype= torch.float32, requires_grad=True)

def forward(x):
    return w*x

def loss(y,y_predicted):
    return ((y - y_predicted)**2).mean()

print(f'Model prediction before  training: {forward(5):.3f}')

learning_rate = 0.01

for epoch in range(55):
    # forward pass
    y_pred = forward(X)
    
    #loss 
    l = loss(Y,y_pred)
    
    # Gradient computation
    l.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    w.grad.zero_()
    
    if epoch%1==0:
        print(f'Epoch {epoch+1},  w {w:.3f},   loss {l:.5f}')
    


print(f'Model after training : {forward(5):.5f}')
print(w)