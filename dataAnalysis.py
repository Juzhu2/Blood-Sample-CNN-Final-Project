# import matplotlib.pyplot as py
# import PIL as Image
# import numpy as np 
import os
import torchvision.datasets as torchImg
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define the list of transformations to be done on image
list_of_transformations = [
    transforms.ToTensor(),
]
# Compose the list of transformations
transform = transforms.Compose(list_of_transformations)

TrainImgs = torchImg.ImageFolder(root='dataset2-master/images/TRAIN', transform=list_of_transformations)

train_loader = DataLoader(TrainImgs, batch_size=64, shuffle=True)

for img, label in train_loader:


    
# with os.open("dataset2-master\\images\\TRAIN\\EOSINOPHIL", 'r') as file:
#     # Read each line in the file
#     for line in file:
#         # Print each line
#         print(line)