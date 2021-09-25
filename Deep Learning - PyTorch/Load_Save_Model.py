# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:06:24 2021

@author: Anshul
"""

import torch 
import torch.nn as nn

# =============================================================================
# One Way To save and load model
# =============================================================================

class Model(nn.Module):
    def __init__(self, in_features):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_features, 1)
        
        
    def forward(self,x):
        x = nn.Sigmoid(self.l1(x))
        return x


model = Model(in_features=6)

for param in model.parameters():
    print(param)

FILE = "model.pth"
#torch.save(model, FILE)

#model = torch.load(FILE)
#model.eval()

#for param in model.parameters():
    #print(param)
    
print(model.state_dict()) #Print Weight and bias for each layer

# =============================================================================
# Second way to load and save model
# =============================================================================

FILE = "model2.pth"
# torch.save(model.state_dict(), FILE)
loaded_model = Model(in_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in model.parameters():
    print(param)

# =============================================================================
# Third way
# =============================================================================

learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
print(optimizer.state_dict())


checkpoint ={
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),}

torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")

epoch = loaded_checkpoint["epoch"]

model = Model(in_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr = 0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())
