# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:35:42 2021

@author: Anshul
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# (0) Prepare dataset

bc = datasets.load_breast_cancer()
x_numpy, y_numpy = bc.data, bc.target
print(x_numpy.shape, y_numpy.shape)

X_train, X_test, y_train,y_test = train_test_split(x_numpy,y_numpy,test_size= 0.2,random_state= 12)

##Scaling data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test= torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)


n_features = X_train.shape[1]
o_features = y_train.shape[1]

print(n_features, o_features)

# (1) Define model
class Logistic_regression(nn.Module):
    def __init__(self, input_size, out_features):
        super(Logistic_regression,self).__init__()
        
        self.lin = nn.Linear(input_size, out_features)
        self.sig = nn.Sigmoid()
        
    def forward(self,x):
        x = self.lin(x)
        x = self.sig(x)
        return x
    

model = Logistic_regression(n_features, o_features)
# (2) Loss and Optimizer
learning_rate = 0.05
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# (3) Training Loop 
n_iter = 400
for epoch in range(n_iter):
    #Forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred,y_train)
    
    #backward pass
    loss.backward()
    
    #update parameters
    optimizer.step()
    
    optimizer.zero_grad()
    
    [w,b] = model.parameters()
    
    if epoch%10 == 0 :
        print(f'Epoch: {epoch+1}, W : {w[0][0]:.3f}, Loss : {loss:.5f}')
        


with torch.no_grad():
    prediction = model(X_test)
    prediction1 = prediction.round() # If x<0.5 --> 0 otherwise 1
    
    acc = prediction1.eq(y_test).sum() / float(y_test.shape[0])
    
    print(f'Accuracy of Model : {acc:.3f}')
    

print(w)
    
    