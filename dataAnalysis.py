import matplotlib.pyplot as py
from PIL import Image
import numpy as np 
import os
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torch.utils.data import DataLoader
from torchvision.transforms import v2

#############################################
# False to not show images. True otherwise.
showImgs = False
#############################################

list_o_transformaiton = v2.Compose([
    v2.ToTensor()
])

# Make ImageFolderClasses for all 3 dataset folders --> (TEST)(TEST_SIMPLE)(TRAIN)
# PlotImg: for Matplotlib Visual Inspection
testSimplePlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE")
trainPlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN")
testPlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST")

testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformaiton)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformaiton)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformaiton)

# number of rows and columns in Test_Simple_Imgs
SIMPLE_NUM_ROWS = 2
SIMPLE_NUM_COLS = 5
# ^^ in Train
TRAIN_NUM_ROWS = 10
TRAIN_NUM_COLS = 10
# ^^ in Test
TEST_NUM_ROWS = 10
TEST_NUM_COLS = 10 


# Plot all the images using matplotlib
# in a 10x10 figure
for idx, (image, label) in enumerate(testSimplePlotImgs):
    plot = plt.subplot(SIMPLE_NUM_ROWS, SIMPLE_NUM_COLS, (idx % 10) + 1)
    image = np.array(image)
    plot.imshow(image)
    plot.set_adjustable('datalim', share=True)
    plot.set_xlim(left=0,right=350)
    plot.set_ylim(bottom=0, top=300)
    plot.axis('off')
    
    if ((idx % 10) + 1 == 10):
        plt.tight_layout()
        if(showImgs): plt.show()
        plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.tight_layout()
if(showImgs): plt.show()


for idx, (image, label) in enumerate(trainPlotImgs):
    plot = plt.subplot(TRAIN_NUM_ROWS, TRAIN_NUM_COLS, (idx % 100) + 1)
    image = np.array(image)
    plot.imshow(image)
    plot.set_adjustable('datalim', share=True)
    plot.set_xlim(left=0,right=350)
    plot.set_ylim(bottom=0, top=300)
    plot.axis('off')
    
    if (idx % 100) + 1 == 100:
        plt.tight_layout()
        if(showImgs): plt.show()
        plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.tight_layout()
if(showImgs): plt.show()

for idx, (image, label) in enumerate(testPlotImgs):
    plot = plt.subplot(TRAIN_NUM_ROWS, TRAIN_NUM_COLS, (idx % 100) + 1)
    image = np.array(image)
    plot.imshow(image)
    plot.set_adjustable('datalim', share=True)
    plot.set_xlim(left=0,right=350)
    plot.set_ylim(bottom=0, top=300)
    plot.axis('off')
    
    if (idx % 100) + 1 == 100:
        plt.tight_layout()
        if(showImgs): plt.show()
        plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.tight_layout()
if(showImgs): plt.show()

# Check the shape of all the images and see of they are equal
bool1 = True
shape = 2
for idx, (image, label) in enumerate(testSimpleImgs):
    if(idx == 0):
        shape = image.shape
    # print(f"Image {idx}: Tensor size = {image.shape}")

    if shape != image.shape:
        bool1 = False
        break
    
bool2 = True
shape = 2
for idx, (image, label) in enumerate(TrainImgs):
    if(idx == 0):
        shape = image.shape
    # print(f"Image {idx}: Tensor size = {image.shape}")

    if shape != image.shape:
        bool2 = False
        break
    
bool3 = True
shape = 2
for idx, (image, label) in enumerate(TestImgs):
    if(idx == 0):
        shape = image.shape
    # print(f"Image {idx}: Tensor size = {image.shape}")

    if shape != image.shape:
        bool3 = False
        break

