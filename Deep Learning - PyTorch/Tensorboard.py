# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:12:59 2021

@author: Anshul
"""
#Using Tensorboard
#Code is same as feed forward NN

#Import Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys


#Tensorboard object
Writer = SummaryWriter("runs/mnist2")


#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyper Parameters
input_size = 784 # 28x28 image 
hidden_layers = 200
num_classes = 10
batch_size = 64
num_epochs = 2
learning_rate = 0.005

#Dataset
train_dataset = datasets.MNIST(root = './mnistdata', train= True, transform=transforms.ToTensor(), download= False)
test_dataset = datasets.MNIST(root = './mnistdata', train= False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset= train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size,shuffle=False)


# Check the data
example = iter(train_loader)
sample,label = example.next()
print(sample.shape, label.shape) # (100,1,28,28), (100)

#Plot the data
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(sample[i][0], cmap = 'gray')
#plt.show()


##Visualize grid of images -->  64 images bcz batch_size = 64
img_grid = torchvision.utils.make_grid(sample)
Writer.add_image('mnist_image', img_grid)
Writer.close()
#sys.exit()

#FeedForward Model
class FeedForward(nn.Module):
    def __init__(self, inputs, hidden, classes):
        super(FeedForward, self).__init__()
        
        self.l1 = nn.Linear(inputs, hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden, classes)
        
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

#loss and optimizer
model = FeedForward(input_size, hidden_layers, num_classes)


criterion = nn.CrossEntropyLoss() # Here Multiclass classification, thus we use Cross Entropy loss.
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


# =============================================================================
# ##Tensorboard --
# #To view the Graph of network
# =============================================================================

Writer.add_graph(model, sample.reshape(-1,input_size))
Writer.close()
#sys.exit()

#Training Loop
total_steps = len(train_loader)
running_loss =0.0
running_correct = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1,input_size).to(device) # (100,1,28,28) ---> (100,784)
        labels = labels.to(device)
        
        #Forward
        pred = model(images)
        loss = criterion(pred,labels)
        _, prediction = torch.max(pred, 1)
        #Backward
        optimizer.zero_grad()
        loss.backward()
        
        #Update
        optimizer.step()
        
        running_loss += loss.item()
        running_correct += (prediction == labels).sum().item()
        
        if i%100 ==0:
            print(f'Epochs: {epoch+1}/{num_epochs}, Steps: {i+1}/{total_steps}, loss: {loss:.5f}')
            Writer.add_scalar("Training Loss", running_loss/100, epoch * total_steps + i)
            Writer.add_scalar("Training Accuracy", running_correct/100, epoch * total_steps + i)
            running_loss = 0.0
            running_correct = 0
            Writer.close()


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        
        predict = model(images)
        n_samples += labels.shape[0]
        
        #Values, index
        _, prediction = torch.max(predict, 1)
        
        n_correct += (prediction == labels).sum().item()
        
    acc = 100.0 * n_correct/ n_samples
    print(f'Accuracy : {acc}')
    
