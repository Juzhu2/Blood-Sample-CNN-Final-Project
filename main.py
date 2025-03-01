import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader



list_o_transformaiton = v2.Compose([
    v2.ToTensor(),
])

testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformaiton)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformaiton)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformaiton)


testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True)
trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True)
testData = DataLoader(TestImgs, batch_size=32, shuffle=True)

for idx,(image, label) in enumerate(testSimpleData):
    print(label)
    
# for idx,(image, label) in enumerate(trainingData):
#     print(f'Image: {image}')
#     print(f'Label: {label}')

# for image, label in testData:
#     print(f'Image: {image}')
#     print(f'Label: {label}')