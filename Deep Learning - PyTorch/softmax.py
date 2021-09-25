# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:40:14 2021

@author: Anshul
"""
# =============================================================================
# #Softmax function:
# #   - y = exp(x)/sum(exp(x))
# #   - It converts the values into probabilities whose total sum is 1.
# 
# =============================================================================


import torch
import numpy as np

# =============================================================================
# #From Scratch
# =============================================================================
X = np.array([2.0,1.0,0.1], dtype=np.float32)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

y = softmax(X)

print(y)


# =============================================================================
# #With Inbuilt function
# =============================================================================

X1 = torch.tensor([2.0,1.0,0.1], dtype=torch.float32)
y1 = torch.softmax(X1, dim=0)

print(y1)