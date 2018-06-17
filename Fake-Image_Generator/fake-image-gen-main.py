# -*- coding: utf-8 -*-
"""
Created on Sun May 06 09:57:21 2018

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



# Set the hyper parameters
batchsize = 64 # Set the size of the batch
imagesize = 64 # Set the size of the gererated images to (64 x 64)

# Creating the transformations
'''We create a list of transformations (scaling, tensor conversion, 
normalization) to apply to the input images and make it compatible to the Neural network.'''
transform = transforms.Compose([transforms.Scale(imagesize), transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# instead of transforms.Scale use transforms.Resize

# Loading the Dataset
# Download the training data in the ./data folder and apply the previous transformations on each image.
dataset = datasets.CIFAR10(root = './data', download = True, transform = transform)
# As we have batches of data use dataLoader to get the images of the training set batch by batch.
data_loader = dset.DataLoader(dataset, batch_size = batchsize, shuffle = True, num_workers = 2 )
# num_workers = 2 - we will have two parallel threads that will load the data

# Defining the weights_init function that takes as input 
# a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02) # Defeining weights to the Conv Layer
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02) # defining weigths to the BatchNorm layer
        m.bias.data.fill_(0) # all the bias at the BatchNorm layer will be initialized to Zero
        
# Defining the Generator
# NN.Module contains all the tools that allow us to build our neural network     
class Gen(NN.Module):
    
    def __init__(self): 
        # init function defines the properties of the future objects that will be created from your class
        # self- referes to the future object that will be created
        # Activate the inheritance of the inherited NN.Module:
        super(Gen, self).__init__()
        # meta module - huge module composed of several modules - its a property of the object
        #module - different layers and different connections inside the neural network
        self.main = NN.Sequential(
               NN.ConvTranspose2d(in_channels=100, out_channels= 512, kernel_size = 4,
                                  stride = 1, padding = 0, bias = False), # input sise is 100, output - no. of feature maps
               NN.BatchNorm2d(512),
               NN.ReLU(inplace= True), # Relu rectification to break the linearity for the non linearity of the NN
               # 2nd layer input to this is the oputput from the previous layer
               NN.ConvTranspose2d(in_channels= 512, out_channels= 256, kernel_size = 4,
                                  stride = 2, padding = 1, bias = False),
               NN.BatchNorm2d(256),
               NN.ReLU(inplace= True),
               # 3rd layer input to this is the oputput from the previous layer
               NN.ConvTranspose2d(in_channels= 256, out_channels= 128, kernel_size = 4,
                                  stride = 2, padding = 1, bias = False),
               NN.BatchNorm2d(128),
               NN.ReLU(inplace= True),
               # 4th layer input to this is the oputput from the previous layer
               NN.ConvTranspose2d(in_channels= 128, out_channels= 64, kernel_size = 4,
                                  stride = 2, padding = 1, bias = False),
               NN.BatchNorm2d(64),
               NN.ReLU(inplace= True),
               # 5th layer input to this is the oputput from the previous layer 
               NN.ConvTranspose2d(in_channels= 64, out_channels= 3, kernel_size = 4,
                                  stride = 2, padding = 1, bias = False), 
               # out_channels = 3 - corresponding to the three color channels as we need a output of color images
               NN.Tanh() # to break the linearity  and the values to be between -1 to 1, centered around zero
        )

    #we must forward propagate the network
    def forwardprop(self, input): # input - random vector of size 100
        output = self.main(input)
        return output

# out of the Gen class
# create an object of class Gen
Gen_net = Gen()
Gen_net.apply(weights_init) # weights_init(Gen_net)

# Defining the Descriminator
class Des(NN.Module):
    
    def __init__(self):
        super(Des, self).__init__() # Activate the Inheritance
        self.main = NN.Sequential(
                # Takes images as an input and outputs a probability between 0 and 1
                NN.Conv2d(in_channels= 3, out_channels = 64, kernel_size = 4,
                          stride= 2, padding= 1, bias = False),
                NN.LeakyReLU(negative_slope = 0.2, inplace = True), # Convolution of the DEscriminator works better with Leaky Relu
                # Layer 2 - Input of this CONV is the output of the previous one
                NN.Conv2d(in_channels= 64, out_channels = 128, kernel_size = 4,
                          stride= 2, padding= 1, bias = False),
                NN.BatchNorm2d(128),
                NN.LeakyReLU(negative_slope= 0.2, inplace= True),
                # Layer 3 - Input of this CONV is the output of the previous one
                NN.Conv2d(in_channels= 128, out_channels = 256, kernel_size = 4,
                          stride= 2, padding= 1, bias = False),
                NN.BatchNorm2d(256),
                NN.LeakyReLU(negative_slope= 0.2, inplace= True),
                # Layer 4 - Input of this CONV is the output of the previous one
                NN.Conv2d(in_channels= 256, out_channels = 512, kernel_size = 4,
                          stride= 2, padding= 1, bias = False),
                NN.BatchNorm2d(512),
                NN.LeakyReLU(negative_slope= 0.2, inplace= True),
                # Layer 5 - Input of this CONV is the output of the previous one
                #out_channels = 1 - Des outputs a number between 0 and 1
                NN.Conv2d(in_channels= 512, out_channels = 1, kernel_size = 4,
                          stride= 1, padding= 0, bias = False),
                NN.Sigmoid()
            )
                
    #we must forward propagate the network      
    # make the conv output to be in a single view by flattening it to the next fully connected network input
    def forwardprop(self, input):  # input - generated image
        output = self.main(input) # a number between 0 and 1
        return output.view(-1) # After the series of convolutions we must flatten the Conv output
    # flatten - 2D to 1D to make it ready as an input for the Fully connected layers

# out of the Des class
# create an object of class Des
Des_net = Des()
Des_net.apply(weights_init)    

# Training the DCGANS
# two steps:
# 1 - Updating the weights of the Descriminator 
    # 1. train it by giving the real images and set the target to 1
    # 2. train it by giving the fake images and set the target to zero.
    
# 2 - Updating the weights of the Generator
    # 1. generate a fake image and feed it to the descriminator get an output between 0 and 1.
    # 2. Set the new target to 1 in the Des and calculate the loss and back propogate to the Gen.
    # 3. Apply Stochastic Gradient descent and update the weights of the Gen.
    
criterion = NN.BCELoss() # Error calculator - target should be between 0 and 1
#Optimizer for the Descriminator
optimizer_Des = optim.Adam(Des_net.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # Adam - Highly advanced optimiyer for Stochastic gradient descent
# Optimizer for the Generator
optimizer_Gen = optim.Adam(Gen_net.parameters(), lr = 0.0002, betas = (0.5, 0.999)) # Adam - Highly advanced optimiyer for Stochastic gradient descent

# Epochs - number of Iterations - it will go through all the images of the dataset 25 times
for epoch in range(25):
    for i, data in enumerate(data_loader, 0): #
        
        # data has every batch of the data
        # data_loader loads the data
        # I is initialised to 0
        # 1st Step: Updating the weights of the neural network of the discriminator
        Des_net.zero_grad() # initialize the gradients of the Descriminator to zero w.r.t the weights
             
        # Training the discriminator with a real image of the dataset
        real, _ = data # input
        # pytorch accepts only the Torch Variable(Tensors and Gradient) so convert real to pytorch variable
        input = Variable(real)
        target = Variable(torch.ones(input.size()[0])) # setting the target to 1 for all the real images
        output = Des_net(input)
        error_Des_real = criterion(output, target)
        
        # Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # create a random input vector
        fake = Gen_net(noise) # generate the fake images from the generator
        target = Variable(torch.zeros(input.size()[0])) # set the target to zero for fake images
        output = Des_net(fake.detach()) # detach the gradients only the tensor input is required for the Des
        error_Des_fake = criterion(output, target) # calculate the loss between fake and the target
        
        # Backpropagating the total error
        error_Des = error_Des_real + error_Des_fake # total error from both the trainings
        error_Des.backward() # Backpropagation enabled
        optimizer_Des.step() # step function applies the optimizer on the NN of the Des to update the weights based on the total error
        
        # 2nd Step: Updating the weights of the neural network of the generator
        
        Gen_net.zero_grad() # Initialize the weights of the gradients to zero
        target = Variable(torch.ones(input.size()[0])) # we want the Gen to produce more real images hence the target is set to one.
        output = Des_net(fake) # we want to keep the gradinet here so as to update the weights of the Gen
        error_Gen = criterion(output, target) # as we are training the Gen we need to calcualte the error w.r.t Gen
        error_Gen.backward() # Back propagation for Gen Enabled
        optimizer_Gen.step() # step function applies the optimizer on the NN of the Gen to update the weights based on the total error
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps
        print('[%d/%d][%d/%d] Loss_Des: %.4f Loss_Gen: %.4f' % (epoch, 25, i, len(data_loader), error_Des.data[0], error_Gen.data[0]))
        if i % 100 == 0: # for every 100 steps
            vutils.save_image(real, '%s/real_sample.png' % "./results", normalize= True)
            fake = Gen_net(noise)
            vutils.save_image(fake.data, '%s/fake_sample_epoch%03d.png' % ("./results", epoch), normalize= True)
                
            
    
    
