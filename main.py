import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader


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