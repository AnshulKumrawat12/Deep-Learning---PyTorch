# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:51:11 2021

@author: Anshul
"""
# =============================================================================
# #Cross Entropy Loss
# #   # -sum(y*log(y_pred))
# #   # If value is large - Larger loss
# #   # If value is small - Smaller loss
# =============================================================================



# =============================================================================
# #From Scratch
# =============================================================================


import torch
import torch.nn as nn
import numpy as np

Y = np.array([1, 0, 0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.3, 0.6, 0.1])

def CrossEntropy(y,y_pred):
    loss = -np.sum(y* np.log(y_pred))
    return loss

l1 = CrossEntropy(Y, Y_pred_good)
l2 = CrossEntropy(Y, Y_pred_bad)


print(l1)
print(l2)



# =============================================================================
# #Inbuilt Function
# # Inbuilt function applies softmax internally. So no need to apply softmax in NN layer
# =============================================================================


loss1 = nn.CrossEntropyLoss()

Y1 = torch.tensor([0]) # No need to be one hot encoded. It means index 0 has value 1,remaining all 0's.

#Nsamplesxnclasses = 1x3
Y_pred_good1 = torch.tensor([[2.0,1.0,0.1]])
Y_pred_bad1 = torch.tensor([[0.5,2.0,0.3]])

l11 = loss1(Y_pred_good1, Y1)
l22 = loss1(Y_pred_bad1,Y1)

print(l11.item())
print(l22.item())


_,prediction1 = torch.max(Y_pred_good1,1) #It returns the index of max value
_, prediction2 = torch.max(Y_pred_bad1,1)

print(prediction1)
print(prediction2)



# =============================================================================
# #Example with 3 samples
# =============================================================================


Y11 = torch.tensor([2,0,1]) # No need to be one hot encoded. It means index 0 has value 1,remaining all 0's.

#Nsamplesxnclasses = 3x3
Y_pred_good22 = torch.tensor([[1.1,0.01,3.0],[2.0,1.0,0.1],[1.0,3.0,0.1]])
Y_pred_bad22 = torch.tensor([[0.5,2.0,0.3],[1.0,5.0,0.1],[2.0,1.0,0.1]])


l111 = loss1(Y_pred_good22, Y11)
l222 = loss1(Y_pred_bad22,Y11)

print(l111.item())
print(l222.item())


_,prediction11 = torch.max(Y_pred_good22,1) #It returns the index of max value
_, prediction22 = torch.max(Y_pred_bad22,1)

print(prediction11)
print(prediction22)
