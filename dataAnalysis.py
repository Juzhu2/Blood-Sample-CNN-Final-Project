import numpy as np 
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2

#############################################
# False to not show images. True otherwise.
showImgs = True
#############################################

# rng = np.random.default_rng()
# random_int = rng.integers(0, 10)
# if random_int % 2 == 0:
#     random_int += 1

list_o_transformation = v2.Compose([
    v2.ToTensor(),
    v2.GaussianBlur(kernel_size=5, sigma=(0.0001, 10)), 
    v2.RandomGrayscale(p=0.2),
    v2.ColorJitter(brightness=(0.9, 1.1)), 
    v2.ToPILImage()
])

# Make ImageFolderClasses for all 3 dataset folders --> (TEST)(TEST_SIMPLE)(TRAIN)
# PlotImg: for Matplotlib Visual Inspection
testSimplePlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformation)
trainPlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformation)
testPlotImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform= list_o_transformation)

testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformation)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformation)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformation)

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
if showImgs:
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
            plt.show()
            plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
    plt.tight_layout()
    plt.show()


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
            plt.show()
            plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
    plt.tight_layout()
    plt.show()

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
            plt.show()
            plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
    plt.tight_layout()
    plt.show()

# Check the shape of all the images and see of they are equal
# and set bool 1, 2, 3 to false to show there are imgs with unequal shapes
bool1 = False
shape = 2
for idx, (image, label) in enumerate(testSimpleImgs):
    if(idx == 0):
        shape = image.shape
    # print(f"Image {idx}: Tensor size = {image.shape}")

    if shape != image.shape:
        bool1 = True
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

# Print which dataset is of the shape listed and which (if any) 
# have imgs with unequal shapes
if (bool1):
    print(f"Test Simple Imgs is of shape: {shape}")
else:
    print("Test Simple does not have a valid shape")
if (bool2):
    print(f"Train Imgs is of shape: {shape}")
else:
    print("Test Simple does not have a valid shape")
if (bool3):
    print(f"Test Imgs is of shape: {shape}")

