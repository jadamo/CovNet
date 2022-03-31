import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time

# Total number of matrices in the training + test set
N = 1000

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


class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N):
        self.params = torch.zeros([N, 6])
        self.matrices = torch.zeros([N, 100, 100])
        self.N = N

        for i in range(500):
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

def read_training_set(filename, frac_training):
    inputs = torch.zeros([N, 6])
    outputs = torch.zeros([N, 100, 100])

    print(inputs.view)
    for i in range(500):
        file = filename+f'{i:04d}'+".txt"
        #TODO: Convert this to pytorch so it can directly load to GPU
        # load in parameters
        f = open(file)
        header = f.readline()
        header = f.readline()
        f.close()
        header = torch.from_numpy(np.fromstring(header[2:-1], sep=","))
        inputs[i] = torch.cat([header[0:4], header[5:]])
        # load in matrix
        outputs[i] = torch.from_numpy(np.loadtxt(file, skiprows=2))

        #inputs[i] = header
        #outputs[i] = matrix

    # split up the data into a training and test set
    # TODO: once I have more data impliment a validation set
    split_idx = int(500 * frac_training)
    train_in = inputs[0:split_idx];   test_in = inputs[split_idx:N]
    train_out = outputs[0:split_idx]; test_out = outputs[split_idx:N]
    return train_in, train_out, test_in, test_out

def train(net, num_epochs, batch_size, F_loss, optimizer, train_loader):

    net.train()
    for epoch in range(num_epochs):
        avg_loss = 0.
        for (batch_idx, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params).view(batch_size, 100, 100)
            loss = F_loss(prediction, matrix)
            print(loss)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
        print("Epoch : {:d}, average loss: {:0.3f}".format(epoch, avg_loss))
    return net

def main():

    NN_Full = Network_Full(6, 100*100)
    # init parameters
    NN_Full.apply(init_normal)

    batch_size, lr, num_epochs = 10, 0.1, 10

    # use MSE loss function
    Loss = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.SGD(NN_Full.parameters(), lr=lr)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research/scripts/Matrix-Emulator/Training-Set/"
    train_data = MatrixDataset(training_dir, 500)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    #train_in, train_out, test_in, test_out = read_training_set(training_dir, 0.8)

    # Train the network!
    t1 = time.time()
    NN_Full = train(NN_Full, num_epochs, batch_size, Loss, optimizer, train_loader)
    t2 = time.time()
    print("Done training!, took {:0.2f} s".format(t2 - t1))

if __name__ == "__main__":
    main()