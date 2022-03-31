from cmath import nan
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time, math
import matplotlib.pyplot as plt

# Total number of matrices in the training + test set
N = 1000

# For now, let's try just using simple fully-connected network
class Network_Full(nn.Module):

    def __init__(self, D_in, D_out):

        super().__init__()
        self.h1 = nn.Linear(D_in, 64)  # Hidden layer
        self.h2 = nn.Linear(64, 256)
        self.h3 = nn.Linear(256, 512)  # Hidden layer
        self.h4 = nn.Linear(512, 1024)  # Hidden layer
        self.h5 = nn.Linear(1024, 5000)  # Hidden layer
        self.h6 = nn.Linear(5000, 5000)  # Hidden layer
        self.out = nn.Linear(5000, D_out)  # Output layer

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
        return self.out(w)


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N):
        self.params = torch.zeros([N, 6])
        self.matrices = torch.zeros([N, 100, 100])
        self.N = N

        for i in range(N):
            file = data_dir+"CovA-"+f'{i:04d}'+".txt"
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

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.params[idx], self.matrices[idx]

# Standard normal distribution
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def train(net, num_epochs, N, batch_size, F_loss, optimizer, train_loader):
    """
    Train the given network
    """
    net.train()

    num_batches = math.ceil(N / batch_size)
    loss_data = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        avg_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params).view(batch_size, 100, 100)
            loss = F_loss(prediction, matrix)
            assert loss != np.nan and loss != np.inf
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()

        # Aggregate loss information
        print("Epoch : {:d}, average loss: {:0.3f}".format(epoch, avg_loss / num_batches))
        loss_data[epoch] = avg_loss / num_batches
    return net, loss_data

def plot_loss(loss, num_epochs, save_dir):

    x = range(num_epochs)
    plt.title("Fully-Connected Network")
    plt.plot(x, loss, marker=".", color="blue")
    plt.xlabel("epoch")
    plt.ylabel("L1 Loss")
    plt.savefig(save_dir+"loss-plot.png")
    plt.show()

def main():

    NN_Full = Network_Full(6, 100*100)
    # init parameters
    NN_Full.apply(init_normal)

    batch_size = 25
    lr = 0.02
    num_epochs = 30

    # use MSE loss function
    Loss = nn.L1Loss()

    # Define the optimizer
    optimizer = torch.optim.Adam(NN_Full.parameters(), lr=lr)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research/scripts/Matrix-Emulator/Training-Set/"
    save_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Plots/"
    train_data = MatrixDataset(training_dir, 500)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    #train_in, train_out, test_in, test_out = read_training_set(training_dir, 0.8)

    # Train the network!
    t1 = time.time()
    NN_Full, loss = train(NN_Full, num_epochs, 500, batch_size, Loss, optimizer, train_loader)
    t2 = time.time()
    print("Done training!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    plot_loss(loss, num_epochs, save_dir)

if __name__ == "__main__":
    main()