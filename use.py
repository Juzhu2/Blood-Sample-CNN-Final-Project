import numpy as np 
# from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
from main import ccnModel

# Labels for when we want to indentify which cell is whic
Labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
image = Image.open("ADD PATH HERE") #Depending on what images you want to use
 

Img_Model = ccnModel()

# Selects the mode you want to use
Img_Model.load_state_dict(torch.load("ADD PATH NAME HERE", weights_only=True))


# Converts our image into a tensor so it can be put into the model
raw_transform = v2.Compose([
    v2.ToTensor(),  # Convert image to tensor (scales values to [0,1])
])

image = raw_transform(image)
# removes as extra dimensions on our photos
image = image.unsqueeze(0)

with torch.no_grad():  # Disable gradient calculation for inference
    prediction = Img_Model(image)

    # Applies softmax on our predition and scales it to be 0 to 1
    softmax = nn.Softmax(dim = 1)
    prediction = softmax(prediction)

    # Puts it into percentage and prints it out
    softmax_output = prediction.numpy() * 100
    for i in range(len(softmax_output)):
        print(f"{Labels[i]}: {softmax_output[i]}%")

    # Gets the maximum out of all of it and chooses the label according to the max and prints out our prediction 
    predicted_class = torch.argmax(prediction, dim=1).item()
    print("Predicted Class:", Labels[predicted_class])