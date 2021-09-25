# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:42:48 2021

@author: Anshul
"""
# =============================================================================
# #using BCE loss
# =============================================================================

import torch
import torch.nn as nn
import numpy as np

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NeuralNet2,self).__init__()
        
        self.l1 = nn.Linear(in_features=input_size, out_features= hidden_layers)
        self.act1 = nn.ReLU()
        self.l2 = nn.Linear( in_features = hidden_layers, out_features= num_classes)
        
    
    def forward(self,x):
        x = self.l1(x)
        x = self.act1(x)
        x = self.l2
        #Sigmoid at the end
        x = torch.sigmoid(x)
        return x                    
    

model = NeuralNet2(input_size = 28*28, hidden_layers = 5, num_classes=3)
criterion = nn.BCELoss() #(applies softmax)
