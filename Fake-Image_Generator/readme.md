Dataset: CIPHAR10

download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html
The dataset will be downloaded in the form of Batches.
Inintially we shall create a class for both the Generator and the Discriminator seperately
Then we create the instances of these classes and use it in our epoch forloop to train the Descriminator to identify both real and fake images and also the Generator in creating an image which will be accepted from the Discriminator.

Generator - De-convolutional network

Descriminator - Convlutional network

Framework: Pytorch
