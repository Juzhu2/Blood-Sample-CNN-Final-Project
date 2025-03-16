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

Labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
image = Image.open("ADD PATH HERE")
device = 'cuda'

Img_Model = ccnModel()

Img_Model.load_state_dict(torch.load("ADD PATH NAME HERE", weights_only=True))



raw_transform = v2.Compose([
    v2.ToTensor(),  # Convert image to tensor (scales values to [0,1])
])

image = raw_transform(image)
image = image.unsqueeze(0).to(device)

with torch.no_grad():  # Disable gradient calculation for inference
    prediction = Img_Model(image)

    softmax = nn.Softmax(dim = 1)
    prediction = softmax(prediction)

    softmax_output = prediction.cpu().numpy() * 100
    for i in range(len(softmax_output)):
        print(f"{Labels[i]}: {softmax_output[i]}%")


    predicted_class = torch.argmax(prediction, dim=1).item()
    print("Predicted Class:", Labels[predicted_class])