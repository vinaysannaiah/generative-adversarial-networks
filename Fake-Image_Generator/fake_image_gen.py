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
