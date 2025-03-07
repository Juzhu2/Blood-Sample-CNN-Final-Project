import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch import nn


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

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # WRITE CODE HERE: initialize layers, activations
        # This creates 5 layers where it goes from 3 input and passes through the width of 3 for each layer but near the end it goes to 1 width
        self.activation  = nn.ReLU()
        pooling = nn.MaxPool2d(2,2)
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(in_channels=20,out_channels=100, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(in_channels=100,out_channels=75, kernel_size=5, padding= 2)
        self.layer4 = nn.Conv2d(in_channels=75,out_channels=50, kernel_size=5, padding=2) 

        self.linearlayer1 = nn.Linear(240000, 100000)
        self.linearlayer2 = nn.Linear(100000, 10000)
        self.linearlayer3 = nn.Linear(10000, 1000)
        self.linearlayer4 = nn.Linear(1000, 100)
        self.linearlayer5 = nn.Linear(100, 4)

        def forward(self, input):
            # WRITE CODE HERE: pass input through layers and activations and return it
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
            
            partial = partial.flatten(start_dim=1)
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
            


# Loops that prints the index with the img and label shapes.cd
# Test Simple: test loop
for idx,(image, label) in enumerate(testSimpleData):
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')
    
# Train: Test loop
for idx,(image, label) in enumerate(trainingData):
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')

# Test: Test loop
for image, label in testData:
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')