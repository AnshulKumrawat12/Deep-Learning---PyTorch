# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:30:55 2021

@author: Anshul
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torchvision
import math

class WineDataset(Dataset):
    def __init__(self,):
        #data loading
        xy= np.loadtxt('C:\IISc\Online Material\Python Engineer\PyTorch Tutorial\data\wine\wine.csv', delimiter = ",",dtype = np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:, 1:]) #All column except 1st
        self.y = torch.from_numpy(xy[:, [0]]) #Only 1st column
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        #len(dataset)
        return self.n_samples


dataset = WineDataset()
# =============================================================================
# #Printing 1st row of data
# first_data = dataset[0]
# features, label = first_data
# print(features, label)
# 
# =============================================================================

dataloader = DataLoader(dataset = dataset, batch_size= 4, shuffle=True)

#Iterate over data
data_iter = iter(dataloader)
data = data_iter.next()
features1, label1 = data
print(features1, label1)



num_epochs = 2
total_samples = len(dataset)
num_iteration = math.ceil(total_samples/4)
print(total_samples, num_iteration)


for epoch in range(num_epochs):
    for i, (features, label) in enumerate(dataloader):
        if i%5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Steps : {i}/{num_iteration} , {features.shape}, {label.shape} ')