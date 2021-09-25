# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:43:45 2021

@author: Anshul
"""

import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# (0) Prepare Dataset
X_numpy, y_numpy = datasets.make_regression(n_samples= 100, n_features= 1, noise= 20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(y_numpy.astype(np.float32))

n_samples, n_features = X.shape

print(n_features)
print(Y.shape) # It is of size 100 , but we want it as 100x1

# Reshaping Y
Y = Y.view(Y.shape[0],1)
out_features = Y.shape[1]
print(Y.shape)

#Hyperparameters
learning_rate = 0.01
n_iter = 300

# (1) Define Model

model = nn.Linear(n_features, out_features)

#Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

#Training loop
for epoch in range(n_iter):
    #Forward pass and loss
    y_pred = model(X)
    loss = criterion(Y,y_pred)
    
    #Backward pass
    loss.backward()

    #Update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    [w,b] = model.parameters()
    if epoch%10 ==0:
        print(f'Epochs: {epoch+1}, W: {w[0][0]:.3f}, Loss : {loss:.5f}')
        


#Plotting graph
Predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, Predicted, 'b')
plt.show()

