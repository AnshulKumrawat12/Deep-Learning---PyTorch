# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:18:28 2021

@author: Anshul
"""

import torch

eta = torch.tensor(0.1)
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

for epoch in range(50):
    y_hat = w*x
    loss = (y_hat - y)**2
    print(loss)
    
    loss.backward()
    print(loss)
    
    with torch.no_grad():
        w-= eta*w.grad
    
    w.grad.zero_()
    


print(w*50)