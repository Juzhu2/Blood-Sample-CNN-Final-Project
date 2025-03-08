import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import wandb

#This should initalize the wandb so that it can be graph and compared to other runs
run = wandb.init(project="Blood-Sample-Disease-Project", name="Run")
Labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

class ccnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initalizing the acitivation and pooling
        self.activation  = nn.ReLU()
        self.pooling = nn.MaxPool2d(2,2)

        # Convultion layers for CNN
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(in_channels=20,out_channels=100, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(in_channels=100,out_channels=75, kernel_size=5, padding= 2)
        self.layer4 = nn.Conv2d(in_channels=75,out_channels=50, kernel_size=5, padding=2) 

        # This is the linear layer which we would later use after flattening our convolution layers at the end
        self.linearlayer1 = nn.Linear(240000, 100000)
        self.linearlayer2 = nn.Linear(100000, 10000)
        self.linearlayer3 = nn.Linear(10000, 1000)
        self.linearlayer4 = nn.Linear(1000, 100)
        self.linearlayer5 = nn.Linear(100, 4)

    def forward(self, input):
        
        # Runs the images through the convolution layers and activation while also pooling them so the images is in a smaller resolution
        partial = self.layer1(input)
        partial = self.activation(partial)
        partial = self.layer2(partial)
        partial = self.activation(partial)
        partial = self.pooling(partial)

        partial = self.layer3(partial)
        partial = self.activation(partial)
        partial = self.layer4(partial)
        partial = self.activation(partial)
        partial = self.pooling(partial)
        
        # Flattens the convolution layer in which we pass through the linear layer that would output 4 values
        partial = torch.flatten(partial)
        partial = self.linearlayer1(partial)
        partial = self.activation(partial)
        partial = self.linearlayer2(partial)
        partial = self.activation(partial)
        partial = self.linearlayer3(partial)
        partial = self.activation(partial)
        partial = self.linearlayer4(partial)
        partial = self.activation(partial)
        final = self.linearlayer5(partial)


        return final

# List of transformations 
# Gausian Blur 
# Random GrayScale
# ColorJitter
list_o_transformation = v2.Compose([
    v2.ToTensor(),
    v2.GaussianBlur(kernel_size=3, sigma=5), 
    v2.RandomGrayscale(p=0.3),
    v2.ColorJitter(brightness=(0.5, 1.5))   # *  Possibly change the Hue, Saturation, and Contrast
                                            # of the imgs  *
])

# Create out ImageFolder Classes out of our dataset
# Test Simple is a small (~60) set of imgs 
# Train and Test are our actual datasets 
testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformation)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformation)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformation)

# Create the Dataloaders that will go through our dataset.
testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True)
trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True)
testData = DataLoader(TestImgs, batch_size=32, shuffle=True)

            
EPOCH = 10 # Epoch used to run through the model
lr = 0.01 # Learning rate for our optimizer

Img_Model = ccnModel()
lossFunc = nn.CrossEntropyLoss()
optimizer = optim.Adam(Img_Model.parameters(), lr) # create the optimizer -----> ADAM


"""
# Loops that prints the index with the img and label shapes.cd
# Test Simple: test loop
    for idx,(image, label) in enumerate(testSimpleData):
        prediction = Img_Model(image)
        loss = lossFunc(prediction, label)
        
        print(f"loss: {loss}")
        print(f"idx: {idx}")
        print(f'Image: {image.shape}')
        
        loss.backward()                                  
        optimizer.step()                                 
        optimizer.zero_grad()  
"""


# Train: Run through the epoch and batches and also logs it into the wandb
for i in range(EPOCH):
    for idx,(image, label) in enumerate(trainingData):
        prediction = Img_Model(image)
        loss = lossFunc(prediction, label)

        print(f"Epoch:: {EPOCH}   Loss:: {loss}   Image:: {image.shape}")

        run.log({"train loss": loss})
        loss.backward()                                  
        optimizer.step()                                 
        optimizer.zero_grad()  

# Test: Test loop
with torch.no_grad():
    for image, label in testData:
        prediction = Img_Model(image)
        loss = lossFunc(prediction, label)
        
        print(f"loss: {loss}")
        print(f"idx: {idx}")
        print(f'Image: {image.shape}')
    