# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:04:36 2021

@author: Anshul
"""

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)
input_samples, input_features = X.shape
output_features = Y.shape[1]

learning_rate = 0.01
n_iter = 2000

print(input_samples,input_features,output_features)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = self.lin(x)
        return x
        

model = LinearRegression(input_features, output_features)        


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


for epoch in range(n_iter):
    #Forward Pass
    y_pred = model(X)
    
    #Loss and backward pass
    l = loss(Y,y_pred)
    
    l.backward()
    
    #Update parameters
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch%5 == 0:
        [w,b] = model.parameters()
        print("W matrix shape :", w.shape)
        print("Bias shape : ",b.shape)
        print(f'Epochs: {epoch+1}, W: {w[0][0]:.3f}, loss: {l:.5f}')


print(f'Model after training: {model(X_test).item():.3f}')    
        
        
        

    