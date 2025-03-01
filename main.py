import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader



list_o_transformaiton = v2.Compose([
    v2.ToTensor(),
    v2.GaussianBlur(kernel_size=3, sigma=5), 
    v2.RandomGrayscale(p=0.3),
    v2.ColorJitter(brightness=(0.5, 1.5), contrast=1.0, saturation=1.0, hue=1.0)
])

testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformaiton)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformaiton)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformaiton)


testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True)
trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True)
testData = DataLoader(TestImgs, batch_size=32, shuffle=True)

for idx,(image, label) in enumerate(testSimpleData):
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')
    
for idx,(image, label) in enumerate(trainingData):
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')

for image, label in testData:
    print(f"idx: {idx}")
    print(f'Image: {image.shape}')
    print(f'Label: {label.shape}')