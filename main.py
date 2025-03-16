import numpy as np 
# from matplotlib import pyplot as plt
import torchvision.datasets as torchImg
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import wandb
import torch.nn.functional as F

#This should initalize the wandb so that it can be graph and compared to other runs
run = wandb.init(project="Blood-Sample-Disease-Project", name="Run")
Labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

def compute_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)  # Get predicted class indices
    correct = (predicted_classes == labels).sum().item()  # Count correct predictions
    total = labels.size(0)  # Total number of samples
    accuracy = (correct / total) * 100  # Convert to percentage
    return accuracy


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for class imbalance (set to 1 if not needed)
        :param gamma: Focusing parameter (higher = more focus on hard examples)
        :param reduction: 'mean' (default) or 'sum' for loss aggregation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions (logits before softmax for multi-class, probability for binary)
        :param targets: Ground truth labels
        """
        ce_loss = F.cross_entropy(inputs.to(device), targets.to(device), reduction="none")  # Compute standard CE loss
        p_t = torch.exp(-ce_loss)  # Get softmax probabilities
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Apply focal weighting

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ccnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initalizing the acitivation and pooling
        self.activation  = nn.ReLU()
        self.pooling = nn.MaxPool2d(2,2)
    

        # Convultion layers for CNN
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(in_channels=10,out_channels=50, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(in_channels=50,out_channels=30, kernel_size=5, padding= 2)
        self.layer4 = nn.Conv2d(in_channels=30,out_channels=10, kernel_size=5, padding=2) 


        # This is the linear layer which we would later use after flattening our convolution layers at the end
        self.linearlayer1 = nn.Linear(10 * 60 * 80, 5000)
        self.linearlayer2 = nn.Linear(5000, 3000)
        self.linearlayer3 = nn.Linear(3000, 1000)
        self.linearlayer4 = nn.Linear(1000, 100)
        self.linearlayer5 = nn.Linear(100, 4)
 

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
        final = self.linearlayer5(partial)
      
        
        return final

# List of transformations 
# Gausian Blur 
# Random GrayScale
# ColorJitter
list_o_transformation = v2.Compose([
    v2.ToTensor(),
    v2.GaussianBlur(kernel_size=3, sigma=5), 
    v2.RandomGrayscale(p=0.3),
    v2.ColorJitter(brightness=(0.5, 1.5), saturation =(1, 5), hue=(-0.5, 0.5), contrast=(1,5)),   # *  Possibly change the Hue, Saturation, and Contrast
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
                                            # of the imgs  *
])

# Create out ImageFolder Classes out of our dataset
# Test Simple is a small (~60) set of imgs 
# Train and Test are our actual datasets 
testSimpleImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST_SIMPLE", transform=list_o_transformation)
TrainImgs = torchImg.ImageFolder(root="dataset2-master/images/TRAIN", transform=list_o_transformation)
TestImgs = torchImg.ImageFolder(root="dataset2-master/images/TEST", transform=list_o_transformation)

# Create the Dataloaders that will go through our dataset.
testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True, pin_memory=True)
trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
testData = DataLoader(TestImgs, batch_size=32, shuffle=True, pin_memory=True)


EPOCH = 10 # Epoch used to run through the model
lr = 0.0001 # Learning rate for our optimizer

Img_Model = ccnModel()
device = ''
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu' 

torch.backends.cudnn.benchmark = True  # Speeds up training
torch.cuda.empty_cache()  # Clears unused memory    

Img_Model = Img_Model.to(device)
print(next(Img_Model.parameters()).device)

lossFunc = FocalLoss(alpha=0.25, gamma=2.0)

optimizer = optim.Adam(Img_Model.parameters(), lr=lr)

# Train: Run through the epoch and batches and also logs it into the wandb
for i in range(EPOCH):
    for idx,(image, label) in enumerate(trainingData):
        image = image.to(device, dtype=torch.float32, non_blocking=True)
        label = label.to(device, dtype=torch.long, non_blocking=True)

        prediction = Img_Model(image)
        loss = lossFunc(prediction, label)

        accuracy = compute_accuracy(prediction, label)

        print(f"Epoch:: {i}   Loss:: {loss}   Accuracy:: {accuracy}%  Image:: {image.shape}")

        run.log({"train loss": loss, "index": idx})
        loss.backward()                                  
        optimizer.step()                                 
        optimizer.zero_grad() 
 
# # Test: Test loop
# for i in range(EPOCH):
#     for image, label in testData:
#         prediction = Img_Model(image)
#         loss = lossFunc(prediction, label)
#         accuracy = compute_accuracy(prediction, label)

#         print(f"Loss:: {loss}   Loss:: {loss}  Accuracy::{accuracy}  Image:: {image.shape}")
#         run.log({"test loss": loss, "accuracy": accuracy})



# #Comment if you don't want to save
# torch.save(Img_Model.state_dict(), "ADD PATH") #ADD PATH HERE




