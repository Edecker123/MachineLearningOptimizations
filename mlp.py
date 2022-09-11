import torch #first we import the library 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda #first w
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from torch import nn 


trainingData=datasets.FashionMNIST( #here we create a dataset that is iterable 
    root="data", 
    train=True,
    download=True,  # train
    transform=ToTensor()
)

testData=datasets.FashionMNIST( #here we create a testing data set to prevent from overfitting 
    root="data",
    train=False,  # trai
    download=True,  # train
    transform=ToTensor()
)

train_dataloader=DataLoader( #here we create a dataloader, a dataloader wraps an iterable around the dataset for learning
    trainingData,
    batch_size=64,
    shuffle=True
)

test_dataloader=DataLoader( #here we do testdata iterable
    testData,
    batch_size=64,
    shuffle=True
)

trainFeatures, trainLabels=next(iter(train_dataloader))

ds=datasets.FashionMNIST(
    root="data", 
    train=True,  # train
    download=True,  # train
    transform=ToTensor(), 
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))



device ="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten=nn.Flatten() #this will take the image and flatten it into a vector of pixels 
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28, 512), #this applies a linear transformation on the vectors with the indices as weights and biasses
            nn.ReLU(), #this is our activation
            nn.Linear(512, 512),
            nn.ReLU(),   
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

model=NeuralNetwork().to(device) 

X = torch.rand(1, 28, 28, device=device) #chossing one randome image and shoving it through
logits = model(X) #getting the output of the model
pred_probab = nn.Softmax(dim=1)(logits) #scaling to probabilities
y_pred = pred_probab.argmax(1) #picking the max value
print(f"Predicted class: {y_pred}")