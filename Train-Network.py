import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# For now, let's try just using simple fully-connected network
class Network_Full(nn.Module):

    def __init__(self, D_in, D_out):

        super().__init__()
        self.h1 = nn.Linear(D_in, 256)  # Hidden layer
        self.h2 = nn.Linear(256, 512)  # Hidden layer
        self.h3 = nn.Linear(512, 1024)  # Hidden layer
        self.h4 = nn.Linear(1024, 5000)  # Hidden layer
        self.out = nn.Linear(5000, D_out)  # Output layer

    # Define the forward propagation of the model
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        w1 = F.relu(self.h1(X))
        w2 = F.relu(self.h2(w1))
        w3 = F.relu(self.h3(w2))
        w4 = F.relu(self.h4(w3))
        return self.out(w4)


# Standard normal distribution
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def read_training_set(filename, frac_training):
    inputs = np.array((6, 1000))
    outputs = np.array((100, 100, 1000))

    for i in range(2):
        file = filename+f'{i:04d}'+".txt"
        #TODO: Convert this to pytorch so it can directly load to GPU
        f = open(file)
        header = f.readline()
        f.close()
        matrix = np.loadtxt(file, skiprows=0)

        #unpack header 
        header = header.split(" = ")

        print(header)

# From d2l chapter 3
def train():
    return 0

def main():

    NN_Full = Network_Full(6, 100*100)
    # init parameters
    NN_Full.apply(init_normal)
    # use MSE loss function
    MSELoss = torch.nn.MSELoss

    batch_size, lr, num_epochs = 10, 0.1, 10
    read_training_set("/home/joeadamo/Research/scripts/Matrix-Emulator/Training-Set/CovA-", 0.8)

if __name__ == "__main__":
    main()