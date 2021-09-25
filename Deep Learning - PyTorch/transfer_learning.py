# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:40:49 2021

@author: Anshul
"""

#Import libraries
import torch
import torch.nn as nn
import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import copy

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameter


#Data transformation
data_transforms = {
    'train': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485,0.486,0.406), std = (0.229,0.224,0.225))]),
    
    'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485,0.486,0.406), std = (0.229,0.224,0.225))]),
        }

#Dataset
data_dir = 'hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x), data_transforms[x]) for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train','val']}

dataset_sizes = {x : len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)


#Model definition
def train_model(model, criterion,optimizer,scheduler,num_epochs =2):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)

        #Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase =='train':
                model.train() #Set model to training mode
            
            else:
                model.eval() #Set model to evaluate mode
                
            
            running_loss = 0.0
            running_corrects = 0
            
            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #Forward
                #Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)
                    
                    #backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
        
                #Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc: .4f}')
            
            #deep copy the model
            if phase == 'val' and epoch_acc> best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model

    
model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs,2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.001)

#Scheduler ---> used for varying step function
step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size = 7, gamma = 0.1)

model = train_model(model, criterion, optimizer, scheduler = step_lr_scheduler, num_epochs=2)
