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
# num_workers = 2 - we will have two parallel threads that will load the data
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size = batchSize, 
                                         shuffle = True,
                                         num_workers = 2)

# Set the hyper parameters
batchsize = 64 # Set the batch size
imagesize = 64 # Set the size of the generated images to (64 x 64)

# Create the transformations
'''We create a list of transformations (scaling, tensor conversion, 
normalization) to apply to the input images and make it compatible to the Neural network.'''
transform = transforms.Compose([transforms.Scale(imagesize), transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# Defining a function that initializes the weights for our Neural Network.
def weights_init(neur_net):
    classname = neur_net.__class.__name__
    if classname.find("Conv") != -1:
        neur_net.weight.data.normal_(0.0, 0.02) # Defeining weights to the Conv Layer
    elif classname.find("BatchNorm") != -1:
        neur_net.weight.data.normal_(0.0, 0.02) # defining weigths to the BatchNorm layer
        neur_net.bias.data.fill_(0) # all the bias at the BatchNorm layer will be initialized to Zero

# Defining the Generator
class Gen(NN.Module): # NN.Module contains all the tools that allow us to build our neural network so let us inherit the NN.module class     
    def __init__(self):
        # init function defines the properties of the future objects that will be created from your class
        # self- refers to the future object that will be created
        # Activate the inheritance of the inherited NN.Module:
        super(Gen, self).__init__()
    
  # We must forward propagate the network incase of the Generator in GAN's
  def forwardprop(self, input): # input - random vector of size 100
    output = self.main(input)
    return output
    
# Defining the Descriminator
class Des(NN.Module): # NN.Module contains all the tools that allow us to build our neural network 
    def __init__(self):
        super(Des, self).__init__() # Activate the Inheritance of the inherited NN.module
        self.main = NN.Sequential(
      # We must define the structure of our neural network in here. Our NN is going to be a sequential network.
        NN.ConvTranspose2d(in_channels=100, out_channels= 512, kernel_size = 4,
                           stride = 1, padding = 0, bias = False), # input size is 100, output - no. of feature maps
        )
  
  #we must forward propagate the network      
  # make the conv output to be in a single view by flattening it to the next fully connected network input
    def forwardprop(self, input):  # input - generated image
        output = self.main(input) # a number between 0 and 1
        return output.view(-1) # After the series of convolutions we must flatten the Conv output
