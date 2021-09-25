# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:59:14 2021

@author: Anshul
"""

import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        print(f'df')
        
PATH = "abc"

# =============================================================================
# #Save on GPU, load on CPU
# =============================================================================

device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

device = torch.device("cpu")
model = model()
model.load_state_dict(torch.load(PATH, map_location=device))


# =============================================================================
# #Save on GPU and load on GPU
# =============================================================================

device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(),PATH)

model=model()
model.load_state_dict(torch.load(PATH))
model.to(device)


#Save on CPU and load on GPU
torch.save(model.state_dict(),PATH)

device = torch.device("cuda")
model=model()
model.load_state_dict(torch.load(PATH, map_location= "cuda:0"))
model.to(device)





