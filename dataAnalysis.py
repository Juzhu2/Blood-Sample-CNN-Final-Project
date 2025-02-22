import matplotlib.pyplot as py
from PIL import Image
import numpy as np 
import os
from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torch.utils.data import DataLoader
from torchvision.transforms import v2


list_o_transformaiton = v2.Compose([
    v2.ToTensor()
])

plotting_img = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE")
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformaiton)

NUM_ROWS = 10
NUM_COLS = 10

for idx, (image, label) in enumerate(plotting_img):
 

    plt1 = plt.subplot(NUM_ROWS, NUM_COLS, idx + 1)

    image = np.array(image)

    plt1.imshow(image)
    plt1.set_title(label)
    plt1.axis('off')


# plt.tight_layout()
plt.show()
