import torch #first we import the library 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda #first w
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from torch import nn 
import torchvision as models
import time



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
            # nn.ReLU(),   
            # nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits
# model=torch.load('model.pth')
model=NeuralNetwork()

learningRate=1e-3
batchSize=64
epochs=0

loss_fn=nn.CrossEntropyLoss() #a mix of MSE and neglog
optimizer=torch.optim.SGD(model.parameters(), lr=learningRate) #optimizing the parameters using stochastic gradient decent that are in the nn 


def trainLoop(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset) #this is the length o f the iterable dataset
    for batch, (X, y) in enumerate(dataloader): #iterate through the input and proper output pairs in our batch 
        pred=model(X) #make a prediction then calculate how wrong we were
        loss=loss_fn(pred, y) #predicting loss
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() #make a step 
        
        if batch % 100==0:
            loss, current=loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct=0,0
    
    with torch.no_grad():
        for X,y in dataloader:
            pred=model(X)
            test_loss+=loss_fn(pred, y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()
        
    test_loss/=num_batches
    correct/=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
  
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    trainLoop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

start=time.time()
jaclist=[]
for batch, (X, y) in enumerate(train_dataloader): 
    # for i in range(0, 64):
        # j=torch.autograd.functional.jacobian(model, X[i])
        # jaclist.append(j)
    j=torch.autograd.functional.jacobian(model, X)
    break
end=time.time()
print(end-start)
torch.save(model, 'model.pth')