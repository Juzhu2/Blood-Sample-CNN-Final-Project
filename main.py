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
import torch.optim.lr_scheduler as lr_scheduler

# This helps us compute what our accruacy for every batch it runs
def compute_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)  # Get predicted class indices
    correct = (predicted_classes == labels).sum().item()  # Count correct predictions
    total = labels.size(0)  # Total number of samples
    accuracy = (correct / total) * 100  # Convert to percentage
    return accuracy

#This is a variation of Cross Entropy Loss which uses Cross Entropy Loss but multiplies it by alpha and exponential by gamma
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs.to(device), targets.to(device), reduction="none")  # Compute standard CE loss
        p_t = torch.exp(-ce_loss)  # Get softmax probabilities
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Apply focal weighting

        # You can set depending on who you what your focal loss to turn out. Either mean or sum
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
        self.dropout = nn.Dropout(0.2)


        # Convultion layers for CNN
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(in_channels=16,out_channels=32, kernel_size=5, padding=2)
        self.layer3 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size=5, padding= 2)
        self.layer4 = nn.Conv2d(in_channels=64,out_channels=2, kernel_size=5, padding=2)

        # This is the linear layer which we would later use after flattening our convolution layers at the end
        self.linearlayer1 = nn.Linear(2 * 60 * 80, 1024)  
        self.linearlayer2 = nn.Linear(1024, 512)
        self.linearlayer3 = nn.Linear(512, 256)
        self.linearlayer4 = nn.Linear(256, 100)
        self.linearlayer5 = nn.Linear(100, 4)
 
    def forward(self, input):
       
        # Runs the images through the convolution layers and activation while also pooling them so the images is in a smaller resolution
        partial = self.layer1(input)
        partial = self.activation(partial)
        partial = self.layer2(partial)
        partial = self.activation(partial)
        partial = self.pooling(partial)
        partial = self.dropout(partial)
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
        partial = self.dropout(partial)
        final = self.linearlayer5(partial)
     
       
        return final


if __name__ == "__main__":
    #This should initalize the wandb so that it can be graph and compared to other runs
    run = wandb.init(project="Blood-Sample-Disease-Project", name="Run")
    Labels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


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


    # Create the Dataloaders that will go through our dataset. (32 batches)
    testSimpleData = DataLoader(testSimpleImgs, batch_size=32, shuffle=True, pin_memory=True)
    trainingData = DataLoader(TrainImgs, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
    testData = DataLoader(TestImgs, batch_size=32, shuffle=True, pin_memory=True)

    EPOCH = 18 # Epoch used to run through the model

    lr = 3e-5# Learning rate for our optimizer

    # This switches between cuba(GPU) or CPU depending if our GPU is avaliable or not
    Img_Model = ccnModel()
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # This helps our model run faster
    torch.backends.cudnn.benchmark = True  # Speeds up training
    torch.cuda.empty_cache()  # Clears unused memory    

    # This is a testbench to see what our model is actually using before it acutally runs the code
    Img_Model = Img_Model.to(device)
    print(next(Img_Model.parameters()).device)

    lossFunc = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(Img_Model.parameters(), lr=lr, weight_decay=0.0001)
    # scheduler = lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.5, total_iters=20)

    # Train: Run through the epoch and batches and also logs it into the wandb
    for i in range(EPOCH):
        for idx,(image, label) in enumerate(trainingData):
            # Allows for our data to be passed from CPU to GPU to be computed faster
            image = image.to(device, dtype=torch.float32, non_blocking=True)
            label = label.to(device, dtype=torch.long, non_blocking=True)

            prediction = Img_Model(image)
            loss = lossFunc(prediction, label)

            # Calculates our accuracy using the accuracy fucntion created earlier
            accuracy = compute_accuracy(prediction, label)

            # Prints our data into termnial to see how our model runs
            print(f"Epoch:: {i}   Loss:: {loss}   Accuracy:: {accuracy}%  Image:: {image.shape}")

            # Uploads data in wandb so it can be tracked and saved
            run.log({"train loss": loss, "train accuracy": accuracy})
            loss.backward()                                  
            optimizer.step()                                
            optimizer.zero_grad()
        # scheduler.step()
   
    # Test: Test if our unseen data matches what we have learned
    with torch.no_grad():
        for i in range(EPOCH):
            for image, label in testData:
                # Allows for our data to be passed from CPU to GPU to be computed faster
                image = image.to(device, dtype=torch.float32, non_blocking=True)
                label = label.to(device, dtype=torch.long, non_blocking=True)

                prediction = Img_Model(image)
                loss = lossFunc(prediction, label)

                # Calculates our accuracy using the accuracy fucntion created earlier
                accuracy = compute_accuracy(prediction, label)

                # Uploads data in wandb so it can be tracked and saved
                print(f"Loss:: {loss}  Accuracy::{accuracy}  Image:: {image.shape}")
                run.log({"test loss": loss, "test accuracy": accuracy})



    # if you don't want to save comment
    PATH = 'PATH'
    torch.save(Img_Model.state_dict(), PATH) #Saves model so it can be used for later cases
   

