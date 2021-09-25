# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:29:52 2021

@author: Anshul
"""
#Import libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets,models,transforms
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import copy




#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Hyper parameters
learning_rate = 0.001




#Transformation
transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.456,0.468,0.5), std=(0.229,0.224,0.225))])


#Dataset
data_dir = 'hymenoptera_data'
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(dataset=train_dataset,batch_size=4,shuffle = True)
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False)



#Dataset Visualisation
dataset_class = train_dataset.classes
print(dataset_class)

example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape)

def imshow(img):
    img = img/2 + 0.5 #Unnormalize
    pimg = img.numpy()
    plt.imshow(np.transpose(pimg,(1,2,0)))
    plt.show()
    
imshow(torchvision.utils.make_grid(samples))



#Model Definition
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict()) #Model weights save
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' *10)
        
        model.train()
        
        running_loss =0.0
        running_correct = 0.0
        
        for image,labels in train_loader:
            image = image.to(device)
            labels = labels.to(device)
            
            #Forward pass
            outputs = model(image)
            _, pred = torch.max(outputs,1)
            loss = criterion(outputs, labels)
            
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #Scheduler Step
            scheduler.step()
            
            running_loss += loss.item() * image.size(0)
            running_correct += torch.sum(pred == labels.data)
            
        epoch_loss = running_loss / len(train_dataset) #len(train_dataset) Gives total images in training
        epoch_accuracy = running_correct/ len(train_dataset)
            
        print(f'Train Loss : {epoch_loss}, Accuracy: {epoch_accuracy}')
            
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model
            
            
def test_model(model,criterion):
    
    model.eval()
    
    n_correct = 0
    n_samples = 0
    running_loss = 0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            predict = model(images)
            _, prediction = torch.max(predict,1)
            loss = criterion(predict, labels)
            
            n_samples += labels.shape[0]
            n_correct += (prediction == labels).sum().item()
            
            running_loss += loss.item() * images.size(0)
            
        val_loss = running_loss / n_samples
        accuracy = 100.0 * n_correct/ n_samples
        print(f'Validation Accuracy: {accuracy}, Loss : {val_loss:.5f}')
        
        #deep copy the model
        if  accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())


    print(f'Best val Acc: {best_acc:4f}')

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model


#Resnet Model
model = models.resnet18(pretrained = True)
n_ftrs = model.fc.in_features

model.fc = nn.Linear(n_ftrs,2)
model = model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

#Scheduler
lr_step_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma=0.1)

#Model call
tr_model = train_model(model, criterion, optimizer, scheduler = lr_step_scheduler)

tst_model = test_model(model, criterion)


