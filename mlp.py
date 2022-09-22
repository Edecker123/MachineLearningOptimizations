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
import functorch
from functorch import jacrev,vjp,vmap,jacfwd

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
        logits=self.linear_relu_stack(x) #passing into network 
        return logits #returning output 
model=torch.load('model.pth')
# model=NeuralNetwork()

learningRate=1e-3
batchSize=64
epochs=0


loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learningRate)


def trainLoop(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader): #keep track of the iterable batch number 
        pred=model(X)
        loss=loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward() #propogate back
        optimizer.step() #make our step 
        
        if batch % 100==0: #this would be our dataset number 
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

def calc(ine):
    return torch.autograd.functional.jacobian(model,ine)


#this will calculate as a whole
print("jacrev", "jacfwd", "autograd", "vjp")
for batch, (X, y) in enumerate(train_dataloader): #keep track of the iterable batch number
    
    # jac2=vmap(torch.autograd.functional.jacobian(model,X,vectorize=True, strategy='forward-mode'))(X)
    # print(jac2.shape)
    start=time.time() 
    jacobian=vmap(jacrev(model))(X)
    end=time.time()
    delta1=end-start
    
    # start=time.time() #autograd does not like vmap
    # print(1)
    # jad=vmap(calc)
    # X.requires_grad_(True)
    # print(jad(X))
    # end=time.time()
    
    start=time.time()
    jacobian=vmap(jacfwd(model))(X)
    end=time.time()
    delta2=end-start

    start=time.time()
    jaclist=[]
    for i in range(0, len(X)):
        j=torch.autograd.functional.jacobian(model, X[i])
        jaclist.append(j)
    
    jaclist=tuple(jaclist)
    torch.stack(jaclist)
    end=time.time()
    
    
    delta3=end-start
    
    start=time.time()
    jvpprod=vjp(model,X) #64X10
    end=time.time()
    delta4=end-start
    
    print(delta1, delta2 , delta3 , delta4)
    
    # t2=ja[0][0][0][0] 
    
    # assert torch.allclose(t1, t2)

    