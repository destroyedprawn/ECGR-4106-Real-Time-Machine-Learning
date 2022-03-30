#!/usr/bin/env python
# coding: utf-8

# # Problem 1
# a. Build a Convolutional Neural Network, like what we built in lectures (without skip connections), to classify the images across all 10 classes in CIFAR 10. You need to adjust the fully connected layer at the end properly with respect to the number of output classes. Train your network for 300 epochs. Report your training time, training loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare them against a fully connected network (homework 2) on training time, achieved accuracy, and model size. Make sure to submit your code by providing the GitHub URL of your course repository for this course.

# In[ ]:


# Import Libraries
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
data_path = '../data-unversioned/p1ch7/'
from torchvision import transforms

device= (torch.device('cuda') if torch.cuda.is_available()
         else torch.device('cpu'))
print(f"Training on device {device}.")

# Preprocess images
preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

# Store validation and training data, preprocess data
cifar10 = datasets.CIFAR10(data_path, train=True, download=True, transform = preprocess)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True, transform = preprocess)

train_loader = torch.utils.data.DataLoader(cifar10, batch_size=256, shuffle = True)
val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=256, shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(8 * 8 * 16, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 16)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[91]:


import datetime

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader):
# Iterates through each epoch
    model.train()
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for imgs, labels in train_loader:

    # Calculate training loss
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            train_loss = loss_fn(outputs, labels)
    
    # Zeros gradient for backpropagation and gradient calculation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
    
    # Calculate validation loss and accuracy
        correct = 0
        count = 0
        
    # Put model in eval mode
        model.eval()
        
    # Disables gradients for validation
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                batch_size = imgs.shape[0]
                outputs = model(imgs)
                _, label_p = torch.max(outputs, dim=1)
                count += labels.shape[0]
                correct += int((label_p == labels).sum())
                val_loss = loss_fn(outputs, labels)
    
    # Calculate accuracy
        acc = correct/count
        
        if epoch == 1 or epoch % 20 == 0:
            print('{} Epoch: {}, Training Loss: {}, Validation Loss {}, Accuracy {}'.format(
                datetime.datetime.now(), epoch, float(train_loss), float(val_loss), acc))
    


# In[3]:


learning_rate = 1e-2
model = Net()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# b. Extend your CNN by adding one more additional convolution layer followed by an activation function and pooling function. You also need to adjust your fully connected layer properly with respect to intermediate feature dimensions. Train your network for 300 epochs. Report your training time, loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare your model size and accuracy over the baseline implementation in Problem1.a. Do you see any over-fitting? Make sure to submit your code by providing the GitHub URL of your course repository for this course.

# In[92]:


class ExtendedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size = 3, padding = 1)
        self.fc1 = nn.Linear(4 * 4 * 8, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
        out = out.view(-1, 4 * 4 * 8)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[93]:


learning_rate = 1e-2
model = ExtendedNet()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# # Problem 2
# a. Build a ResNet based Convolutional Neural Network, like what we built in lectures (with skip connections), to classify the images across all 10 classes in CIFAR 10. For this problem, let’s use 10 blocks for ResNet and call it ResNet-10. Use the similar dimensions and channels as we need in lectures. Train your network for 300 epochs. Report your training time, training loss, and evaluation accuracy after 300 epochs. Analyze your results in your report and compare them against problem 1.b on training time, achieved accuracy, and model size. Make sure to submit your code by providing the GitHub URL of your course repository for this course.

# In[49]:


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = torch.relu(out)
        return out + x


# In[50]:


class ResNetDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1, bias=True)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans = n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        


# In[51]:


learning_rate = 3e-3
model = ResNetDeep()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# b. Develop three additional trainings and evaluations for your ResNet-10 to assess the impacts of regularization to your 
# ResNet-10.
# 
# * Weight Decay with lambda of 0.001
# * Dropout with p=0.3
# * Batch Normalization
# Report and compare your training time, training loss, and evaluation accuracy after 300 epochs across these three different trainings. Analyze your results in your report and compare them against problem 1.a on training time, achieved accuracy

# In[53]:


# 1. Weight Decay with lambda of 0.001
learning_rate = 3e-3
model = ResNetDeep()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001) # Using optimizer param for weight decay
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# In[56]:


# 2. Dropout with p=0.3
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_chans, n_chans, kernel_size=3, padding=1, bias=True)
        self.conv1_dropout = nn.Dropout2d(p=0.3)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_dropout(out)
        out = torch.relu(out)
        return out + x
    
class ResNetDeepDropout(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1, bias=True)  
        self.conv1_dropout = nn.Dropout2d(p=0.3)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans = n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[57]:


learning_rate = 3e-3
model = ResNetDeepDropout()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# In[59]:


# 3. Batch Normalization Report
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                             padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                             nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x
    
class ResNetDeepBatchNorm(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=10):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1, bias=False)  # No bias needed as the Batch Norm handles it
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans = n_chans1)]))
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# In[60]:


learning_rate = 3e-3
model = ResNetDeepBatchNorm()
model = model.to(device=device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001)
loss_fn = nn.CrossEntropyLoss()
n_epochs = 300

training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader)


# # Model Sizes

# In[99]:


from ptflops import get_model_complexity_info

# Regular CNN Complexity
macs, params = get_model_complexity_info(Net(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[100]:


# Extended CNN Complexity    
macs, params = get_model_complexity_info(ExtendedNet(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[98]:


# ResNet based CNN Complexity    
macs, params = get_model_complexity_info(ResNetDeep(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[97]:


# ResNet based CNN Complexity with weight decay regularization   
macs, params = get_model_complexity_info(ResNetDeep(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[95]:


# ResNet based CNN Complexity with dropout regularization  
macs, params = get_model_complexity_info(ResNetDeepDropout(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[94]:


# ResNet based CNN Complexity with batch norm regularization   
macs, params = get_model_complexity_info(ResNetDeepBatchNorm(), (3, 32, 32), as_strings=True,
                                                   print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




