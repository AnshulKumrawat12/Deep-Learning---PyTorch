# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:53:45 2021

@author: Anshul
"""

#import Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 


#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#Hyper parameters
batch_size = 4
num_epochs = 4
learning_rate = 0.001
num_class = 10



#Dataset
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean =(0.5,0.5,0.5), std= (0.5,0.5,0.5))])

train_dataset = datasets.CIFAR10(root= './CIFARdata', train = True, transform=transform, download = True)
test_dataset = datasets.CIFAR10(root='./CIFARdata', train= False, transform=transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle= True)
test_loader = DataLoader(dataset= test_dataset, batch_size=batch_size,shuffle=False)

classes = ('plane','car', 'bird','cat','deer', 'dog', 'frog', 'horse','ship', 'truck')

#show data and plot data
example = iter(train_loader)
samples, labels = example.next()
print(samples.shape, labels.shape)

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()    
    
imshow(torchvision.utils.make_grid(samples))

# =============================================================================
# #Test the shape of Conv layers for giving input to fc layer
# #   - ((Img - Filter + 2Padding)/S) +1
# c1 = nn.Conv2d(3, 5, 3)
# m = nn.MaxPool2d(2,2)
# c2 = nn.Conv2d(5, 15, 3)
# print(samples.shape)
# 
# x = c1(samples)
# print(x.shape)
# x = m(x)
# print(x.shape)
# x = c2(x)
# print(x.shape)
# x = m(x)
# print(x.shape)  # It gives (4,15,6,6)
# =============================================================================









#Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(5, 15, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(15*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,num_classes)

    def forward(self,x):
        x = self.conv1(x) # 1st layer
        x = self.relu(x)
        x = self.pool1(x)
        x = self.pool2(self.relu(self.conv2(x))) #2nd layer
        x = x.reshape(-1,15*6*6) #reshaping x to give input as 15x6x6 to fc layer
        x = self.relu(self.fc1(x))
        x= self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN(num_class)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

#Training loop 
total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        #Forward
        pred = model(images)
        loss = criterion(pred, labels)
        
        #Backward
        loss.backward()
        optimizer.step()    
        optimizer.zero_grad()
        
        if(i+1)%100==0:
            print(f'Epochs: {epoch+1}/{num_epochs}, steps: {i+1}/{total_steps}, loss: {loss:.5f} ')


#Model evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_correct_list = [0 for i in range(10)]
    n_samples_list = [0 for i in range(10)]
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        pred = model(images)
        _,prediction = torch.max(pred,1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            prd = prediction[i]
            if(prd==label):
                n_correct_list[label] += 1
            n_samples_list[label] +=1
        
    acc = 100.0* n_correct/ n_samples
    print(f'Accuracy of Model: {acc}')
    
    for i in range(10):
        x = n_correct_list[i]/n_samples_list[i]
        
        print(f'Accuracy of class{[i]} is {x}')