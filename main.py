import numpy as np 
# from matplotlib import pyplot as plt
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

def compute_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)  # Get predicted class indices
    correct = (predicted_classes == labels).sum().item()  # Count correct predictions
    total = labels.size(0)  # Total number of samples
    accuracy = (correct / total) * 100  # Convert to percentage
    return accuracy


class ccnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initalizing the acitivation and pooling
        self.activation  = nn.ReLU()
        self.pooling = nn.MaxPool2d(2,2)
    

        # Convultion layers for CNN
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(in_channels=10,out_channels=50, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(in_channels=50,out_channels=100, kernel_size=5, padding= 2)
        self.layer4 = nn.Conv2d(in_channels=100,out_channels=200, kernel_size=5, padding=2) 
        self.layer5 = nn.Conv2d(in_channels=200,out_channels=300, kernel_size=5, padding=2) 
        self.layer6 = nn.Conv2d(in_channels=300,out_channels=100, kernel_size=5, padding=2) 

        # This is the linear layer which we would later use after flattening our convolution layers at the end
        self.linearlayer1 = nn.Linear(100 * 30 * 40, 5000)
        self.linearlayer2 = nn.Linear(5000, 3000)
        self.linearlayer3 = nn.Linear(3000, 2000)
        self.linearlayer4 = nn.Linear(2000, 1000)
        self.linearlayer5 = nn.Linear(1000, 500)
        self.linearlayer6 = nn.Linear(500, 250)
        self.linearlayer7 = nn.Linear(250, 70)
        self.linearlayer8 = nn.Linear(70, 4)

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
        partial = self.layer5(partial)
        partial = self.activation(partial)
        partial = self.layer6(partial)
        partial = self.activation(partial)
        partial = self.pooling(partial)
        
        # Flattens the convolution layer in which we pass through the linear layer that would output 4 values
        partial = torch.flatten(partial, start_dim=1)
        partial = self.linearlayer1(partial)
        partial = self.activation(partial)
        partial = self.linearlayer2(partial)
        partial = self.activation(partial)
        partial = self.linearlayer3(partial)
        partial = self.activation(partial)
        partial = self.linearlayer4(partial)
        partial = self.activation(partial)
        partial = self.linearlayer5(partial)
        partial = self.activation(partial)
        partial = self.linearlayer6(partial)
        partial = self.activation(partial)
        partial = self.linearlayer7(partial)
        partial = self.activation(partial)
        final = self.linearlayer8(partial)
        
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
testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformation,)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformation)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformation)

# Create the Dataloaders that will go through our dataset.
testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True)
trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True)
testData = DataLoader(TestImgs, batch_size=32, shuffle=True)

EPOCH = 100 # Epoch used to run through the model
lr = 0.001 # Learning rate for our optimizer

Img_Model = ccnModel()
device = ''
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

Img_Model.to(device)

lossFunc = nn.CrossEntropyLoss()

optimizer = optim.Adam(Img_Model.parameters(), lr=lr)


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
        image = image.to(device)
        label = label.to(device)
        prediction = Img_Model(image)
        loss = lossFunc(prediction, label)

        accuracy = compute_accuracy(prediction, label)

        print(f"Epoch:: {i}   Loss:: {loss}   Accuracy:: {accuracy}%  Image:: {image.shape}")

        run.log({"train loss": loss, "index": idx})
        loss.backward()                                  
        optimizer.step()                                 
        optimizer.zero_grad() 
 

'''
# Test: Test loop
for image, label in testData:
    prediction = Img_Model(image)
    loss = lossFunc(prediction, label)
    accuracy = compute_accuracy(prediction, label)

    print(f"Loss:: {loss}   Accuracy::{accuracy}  Image:: {image.shape}")
    run.log({"test loss": loss})
    
'''