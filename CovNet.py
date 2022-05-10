import torch
import torch.nn as nn
from torch.nn import functional as F
from os.path import exists
import numpy as np

# For now, let's try just using simple fully-connected network
class Network_Full(nn.Module):

    def __init__(self, D_in, D_out):

        super().__init__()
        self.h1 = nn.Linear(D_in, 64)  # Hidden layer
        self.h2 = nn.Linear(64, 128)
        self.h3 = nn.Linear(128, 256)
        self.h4 = nn.Linear(256, 512)  # Hidden layer
        self.h5 = nn.Linear(512, 1024)  # Hidden layer
        self.h6 = nn.Linear(1024, 5000)  # Hidden layer
        self.h7 = nn.Linear(5000, D_out)
        self.out = nn.Linear(D_out, D_out)  # Output layer

    # Define the forward propagation of the model
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        w = F.relu(self.h1(X))
        w = F.relu(self.h2(w))
        w = F.relu(self.h3(w))
        w = F.relu(self.h4(w))
        w = F.relu(self.h5(w))
        w = F.relu(self.h6(w))
        w = F.relu(self.h7(w))
        return self.out(w)

"""
Right now this network is pretty much a reverse of VGG, with the exception that there isn't
any "un-pooling" layers.
"""
class Network_ReverseVGG(nn.Module):

    def __init__(self, D_in):

        super().__init__()
        self.f1 = nn.Linear(D_in, 50)
        self.f2 = nn.Linear(50, 100) 
        self.f3 = nn.Linear(100, 100)  # 10x10

        self.C1_1 = nn.ConvTranspose2d(1, 1, kernel_size=3) #12x12
        self.C1_2 = nn.ConvTranspose2d(1, 1, kernel_size=3) #14x14
        self.C1_3 = nn.ConvTranspose2d(1, 1, kernel_size=3) #16x16

        self.C2_1 = nn.ConvTranspose2d(1, 1, kernel_size=8) # 23x23
        self.C2_2 = nn.ConvTranspose2d(1, 1, kernel_size=3) # 25x25
        self.C2_3 = nn.ConvTranspose2d(1, 1, kernel_size=3) # 27x27
        self.C2_4 = nn.ConvTranspose2d(1, 1, kernel_size=3) # 29x29

        self.C3_1 = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=2) #64x64
        self.C3_2 = nn.ConvTranspose2d(1, 1, kernel_size=3) #66x66
        self.C3_3 = nn.ConvTranspose2d(1, 1, kernel_size=3) #68x68
        self.C3_4 = nn.ConvTranspose2d(1, 1, kernel_size=3) #70x70

        self.C4_1 = nn.ConvTranspose2d(1, 1, kernel_size=9) #78x78
        self.C4_2 = nn.ConvTranspose2d(1, 1, kernel_size=4) #81x81
        self.C4_3 = nn.ConvTranspose2d(1, 1, kernel_size=4) #84x84

        self.C5_1 = nn.ConvTranspose2d(1, 1, kernel_size=11) #94x94
        self.C5_2 = nn.ConvTranspose2d(1, 1, kernel_size=4)  #97x97
        self.out  = nn.ConvTranspose2d(1, 1, kernel_size=4) # output layer

    # Define the forward propagation of the model
    def forward(self, X):
        # Work thru the fully connected part first
        w = F.leaky_relu(self.f3(F.leaky_relu(self.f2(F.leaky_relu(self.f1(X))))))
        w = w.view(-1, 1, 10, 10) # reshape into an "image"

        w = F.leaky_relu(self.C1_3(F.leaky_relu(self.C1_2(F.leaky_relu(self.C1_1(w))))))
        w = F.leaky_relu(self.C2_4(F.leaky_relu(self.C2_3(F.leaky_relu(self.C2_2(F.leaky_relu(self.C2_1(w))))))))
        w = F.leaky_relu(self.C3_4(F.leaky_relu(self.C3_3(F.leaky_relu(self.C3_2(F.leaky_relu(self.C3_1(w))))))))
        w = F.leaky_relu(self.C4_3(F.leaky_relu(self.C4_2(F.leaky_relu(self.C4_1(w))))))
        w = F.leaky_relu(self.out(F.leaky_relu(self.C5_2(F.leaky_relu(self.C5_1(w))))))
        return w


# Dataset class to handle making training / validation / test sets
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset, train_log, train_inverse):
        self.params = torch.zeros([N, 6])
        self.matrices = torch.zeros([N, 100, 100])
        self.offset = offset
        self.N = N

        for i in range(N):

            idx = i + offset
            # skip this file if it doesn't exist (handles incomplete training sets)
            #if exists(data_dir+"CovA-"+f'{idx:04d}'+".txt") == False:
            #    continue

            file = data_dir+"CovA-"+f'{idx:05d}'+".txt"
            #TODO: Convert this to pytorch so it can directly load to GPU
            # load in parameters
            f = open(file)
            header = f.readline()
            header = f.readline()
            f.close()
            header = torch.from_numpy(np.fromstring(header[2:-1], sep=","))
            self.params[i] = torch.cat([header[0:4], header[5:]])
            # load in matrix
            self.matrices[i] = torch.from_numpy(np.loadtxt(file, skiprows=2))

            if train_log == True:
                self.matrices[i] = symmetric_log(self.matrices[i])


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.params[idx], self.matrices[idx]


def matrix_loss(prediction, target, norm):
    """
    Custom loss function that includes a penalizing term for non-symmetric outputs
    """
    # normal l-n norm term
    l1 = torch.pow(prediction - target, norm)
    l1 = torch.mean(l1)

    # term that's non-zero for non-symmetric matrices
    asymmetric_predict = prediction - torch.transpose(prediction,1,2)
    l2 = torch.pow(asymmetric_predict, norm)
    l2 = torch.mean(l2)

    return l1 + l2


def symmetric_log(m):
    """
    Takes a a matrix and returns the piece-wise logarithm for post-processing
    sym_log(x) =  log10(x),  x > 0
    sym_log(x) = -log10(-x), x < 0
    """
    pos_m, neg_m = np.ones(m.shape), np.zeros(m.shape)
    pos_idx = np.where(m >= 0)
    neg_idx = np.where(m < 0)
    pos_m[pos_idx] = m[pos_idx]
    neg_m[neg_idx] = m[neg_idx]

    pos_m = np.log10(pos_m)
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -np.log10(-1*neg_m[neg_idx])
    return torch.from_numpy(pos_m + neg_m)

def symmetric_exp(m):
    """
    Takes a matrix and returns the piece-wise exponent
    sym_exp(x) = 10^x,   x > 0
    sym_exp(x) = -10^-x, x < 0
    This is the reverse operation of symmetric_log
    """
    pos_m, neg_m = np.zeros(m.shape), np.zeros(m.shape)
    pos_idx = np.where(m >= 0)
    neg_idx = np.where(m < 0)
    pos_m[pos_idx] = m[pos_idx]
    neg_m[neg_idx] = m[neg_idx]

    pos_m = 10**pos_m
    pos_m[(pos_m == 1)] = 0
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -10**(-1*neg_m[neg_idx])
    return torch.from_numpy(pos_m + neg_m)