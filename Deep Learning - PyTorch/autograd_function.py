# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 11:15:20 2021

@author: Anshul
"""

import torch

# =============================================================================
# 
# ## When we apply backward() at scalar value
# 
# =============================================================================

x1 = torch.randn(3, requires_grad=True)
print(x1) # Print 3 random values

y = x1+2
print(y)

z = y*y*2
print(z)
z = z.mean()
print(z) # Scalar value
z.backward()  ## Calculate dz/dx
print(x1.grad)



# =============================================================================
# 
# ## When we apply backward() at vector values
# 
# =============================================================================

x2 = torch.randn(3, requires_grad=True)
print(x2)
y2 = x2+2
print(y2)

p = y2*y2*2
print(p)

r = torch.tensor([1.0,1.0,1.0], dtype=torch.float32) ## Same size as size of p

p.backward(r) ## Calculate dp/dx
print(x2.grad)


# =============================================================================
# 
# #3 ways to remove the grad_fn from the vectors
# ### (1) x.requires_grad_(False)
# ### (2) with torch.no_grad()
# ### (3) x.detach()
# 
# =============================================================================


a = torch.randn(3, requires_grad=True)
print(a.requires_grad)

#(1) x.requires_grad(False)

a.requires_grad_(False)
print(a.requires_grad)

# (2) x.detach()
b = torch.randn(3, requires_grad=True)
print(b.requires_grad)

c = b.detach()
print(c.requires_grad)

#(3) With torch.no_grad()

d = torch.randn(3, requires_grad=True)
with torch.no_grad():
    e = d+2
    print(e.requires_grad)
    
# =============================================================================
# 
# #Example of weights gradient calculations
# 
# =============================================================================
import torch

## When q.zero_grad() is not called, it sum up the gradient values
weights = torch.ones(3, requires_grad=True)
for epoch in range(3):
    q = (weights**3).sum()
    q.backward() # dq/dweights
    print(weights.grad)
    
    
## When q.zero_grad() is called
weights1 = torch.ones(3, requires_grad=True)
for epoch in range(3):
    n = (weights1**3).sum()
    n.backward() # dq/dweights
    print(weights1.grad)
    
    weights1.grad.zero_()
    