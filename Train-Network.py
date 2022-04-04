from cmath import nan
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time, math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

class Network_TCNN(nn.Module):

    def __init__(self, D_in, batch_size):

        self.batch_size = batch_size
        super().__init__()
        self.h1 = nn.Linear(D_in, 50)
        self.h2 = nn.Linear(50, 100)  # 10x10
        self.TCNN1 = nn.ConvTranspose2d(1, 1, kernel_size=(7,7), stride=2) # 25x25
        self.TCNN2 = nn.ConvTranspose2d(1, 1, kernel_size=(8,8), stride=2) # 56x56
        self.CNN1 = nn.Conv2d(1, 1, kernel_size=(6,6)) # 50x50
        self.TCNN3 = nn.ConvTranspose2d(1, 1, kernel_size=(6,6), stride=2) #106x106
        self.CNN2 = nn.Conv2d(1, 1, kernel_size=(4,4)) #103x103
        self.out = nn.Conv2d(1, 1, kernel_size=(4,4)) # Output layer

    # Define the forward propagation of the model
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        w = F.relu(self.h1(X))
        w = F.relu(self.h2(w))
        w = w.view(-1, 1, 10, 10) # reshape into an "image"
        w = F.relu(self.TCNN1(w))
        w = F.relu(self.TCNN2(w))
        w = F.relu(self.CNN1(w))
        w = F.relu(self.TCNN3(w))
        w = F.relu(self.CNN2(w))
        return self.out(w)


# Dataset class to handle making training / validation / test sets
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset):
        self.params = torch.zeros([N, 6])
        self.matrices = torch.zeros([N, 100, 100])
        self.N = N
        self.offset = offset

        for i in range(N):
            idx = i + offset
            file = data_dir+"CovA-"+f'{idx:04d}'+".txt"
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


def train(net, num_epochs, N_train, N_valid, batch_size, F_loss, optimizer, train_loader, valid_loader):
    """
    Train the given network
    """
    num_batches = math.ceil(N_train / batch_size)
    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params).view(batch_size, 100, 100)
            loss = F_loss(prediction, matrix)
            assert loss != np.nan and loss != np.inf
            avg_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()

        # run through the validation set
        net.eval()
        avg_valid_loss = 0.
        for params, matrix in valid_loader:
            prediction = net(params).view(batch_size, 100, 100)
            loss = F_loss(prediction, matrix)
            avg_valid_loss+= loss.item()

        # Aggregate loss information
        print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}".format(epoch, avg_train_loss / num_batches, avg_valid_loss / math.ceil(N_valid / batch_size)))
        train_loss[epoch] = avg_train_loss / num_batches
        valid_loss[epoch] = avg_valid_loss / math.ceil(N_valid / batch_size)
    return net, train_loss, valid_loss

def plot_loss(train_loss, valid_loss, num_epochs, net, save_dir):

    x = range(num_epochs)
    plt.title("Fully-Connected Network")
    plt.plot(x, train_loss, marker=".", color="blue", label="training set")
    plt.plot(x, valid_loss, color="red", ls="--", label="validation set")
    plt.xlabel("epoch")
    plt.ylim(0, 510000)
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.savefig(save_dir+"loss-plot.png")

    plt.figure()
    params = torch.tensor([0.307115, 67.5, 2e-9, 0.122, 1.94853182918671, -0.5386588802904639])
    matrix = net(params)
    matrix = matrix.view(100,100).detach().numpy()
    plt.imshow(matrix, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=np.amin(matrix), vmax=np.amax(matrix)))
    plt.colorbar()
    plt.savefig(save_dir+"network-matrix.png")

    plt.show()

def main():

    batch_size = 25
    lr = 0.2
    num_epochs = 100

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    NN_Full = Network_Full(6, 100*100)
    NN_Full.apply(init_normal)

    NN_TConvolution = Network_TCNN(6, batch_size)
    NN_TConvolution.apply(init_normal)

    # use MSE loss function
    Loss = nn.L1Loss()

    # Define the optimizer
    optimizer = torch.optim.Adam(NN_Full.parameters(), lr=lr)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research/scripts/Matrix-Emulator/Training-Set/"
    save_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Plots/"
    train_data = MatrixDataset(training_dir, N_train, 0)
    valid_data = MatrixDataset(training_dir, N_valid, N_train)
    test_data = MatrixDataset(training_dir, int(N*0.1), int(N*0.9))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    #train_in, train_out, test_in, test_out = read_training_set(training_dir, 0.8)

    # Train the network!
    t1 = time.time()
    NN_Full, train_loss, valid_loss = train(NN_Full, num_epochs, N_train, N_valid, batch_size, Loss, optimizer, train_loader, valid_loader)
    t2 = time.time()
    print("Done training!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    plot_loss(train_loss, valid_loss, num_epochs, NN_Full, save_dir)

if __name__ == "__main__":
    main()