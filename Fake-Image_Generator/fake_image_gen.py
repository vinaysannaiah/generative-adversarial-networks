# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:57:21 2018

@author: Vinay
"""

# Import all the libraries here
import torch
import torchvision.transforms as transforms # To transform the data as suitable to the NN
import torchvision.datasets as datasets # to get the dataset CIPHAR10
import torch.utils.data as dset # To load the Dataset
import torch.nn as NN # Neural network model of Torch
import torch.nn.parallel
import torch.optim as optim # Optimizer
from torch.autograd import Variable # torch variable convertor
import torchvision.utils as vutils # for Visulization purpose

# Loading the dataset
# We download the training set in the ./data folder and we apply the previous transformations on each image.
dataset = dset.CIFAR10(root = './data', 
                       download = True, 
                       transform = transform) 
# We use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = batchSize, 
                                         shuffle = True, num_workers = 2) 

# Set the hyper parameters
batchsize = 64 # Set the batch size
imagesize = 64 # Set the size of the generated images to (64 x 64)

