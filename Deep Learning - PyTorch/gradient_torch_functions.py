# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 15:43:19 2021

@author: Anshul
"""
#Define model (input size, output size, forward pass, backward pass)
#Construct loss and optimizer
#Training loop
#   - forward pass: Compute y_pred
#   -backward pass: compute gradient
#   -update parameters

import torch
import torch.nn as nn

lr = 0.01
n_iter = 70 

X = torch.tensor([1,2,3,4], dtype = torch.float32)
Y = torch.tensor([2,4,6,8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad= True)

def forward(x):
    return w*x

loss = nn.MSELoss()

optimizer = torch.optim.SGD([w], lr)

print(f'Before training : {forward(5)}')
for epoch in range(n_iter):
    #Forward Pass
    y_pred = forward(X)
    
    #Loss and Optimizer
    l = loss(Y,y_pred)
    
    #backward pass
    l.backward()
    
    #optimizer update parameters
    optimizer.step()
    
    optimizer.zero_grad()
    
    print(f'Epochs: {epoch+1}, W: {w:.3f}, Loss: {l:.5f}')
    
print(f'After training : {forward(5):.3f}')




