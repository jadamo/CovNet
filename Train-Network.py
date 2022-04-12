import torch
import torch.nn as nn
from torch.nn import functional as F
from os.path import exists
import numpy as np
import time, math, sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
from CovNet import Network_Full, Network_ReverseVGG, MatrixDataset

# Total number of matrices in the training + validation + test set
N = 52500

# wether to train using the percision matrix instead
train_inverse = False
# wether to train using the log of the matrix
train_log = True

# Standard normal distribution
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

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
        min_val = 1e30; max_val = -1e10
        min_pre = 1e30; max_pre = -1e10
        for params, matrix in valid_loader:
            prediction = net(params).view(batch_size, 100, 100)
            loss = F_loss(prediction, matrix)
            avg_valid_loss+= loss.item()
            min_pre = min(torch.min(prediction), min_pre)
            max_pre = max(torch.max(prediction), max_pre)
            min_val = min(torch.min(matrix), min_val)
            max_val = max(torch.max(matrix), max_val)

        # Aggregate loss information
        print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}".format(epoch, avg_train_loss / num_batches, avg_valid_loss / math.ceil(N_valid / batch_size)))
        print(" min valid = {:0.3f}, max valid = {:0.3f}".format(min_val, max_val))
        print(" min predict = {:0.3f}, max predict = {:0.3f}".format(min_pre, max_pre))
        train_loss[epoch] = avg_train_loss / num_batches
        valid_loss[epoch] = avg_valid_loss / math.ceil(N_valid / batch_size)
    return net, train_loss, valid_loss

def plot_loss(train_loss, valid_loss, num_epochs, net, save_dir):

    x = range(num_epochs)
    max_y = max(train_loss[1], valid_loss[1] * 1.1)
    plt.title("Fully-Connected Network")
    plt.plot(x, train_loss, color="blue", label="training set")
    plt.plot(x, valid_loss, color="red", ls="--", label="validation set")
    plt.xlabel("epoch")
    plt.ylim(0, max_y)
    plt.ylabel("L1 Loss")
    plt.legend()
    plt.savefig(save_dir+"loss-plot.png")

    plt.figure()
    params = torch.tensor([71.30941428788515, 0.29445535560999114, 0.13842482982125812, 1.7164337170568933e-09, 2.01291521504162, 0.3273341556889142])
    matrix = net(params)
    print(torch.min(matrix), torch.max(matrix))
    matrix = matrix.view(100,100).detach().numpy()
    if train_log == True:
        plt.imshow(matrix, cmap="RdBu")
        #plt.imshow(matrix, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=np.amin(matrix), vmax=np.amax(matrix)))
        cbar = plt.colorbar()
        cbar.set_label("log value")
    else:
        plt.imshow(matrix, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=np.amin(matrix), vmax=np.amax(matrix)))
        cbar = plt.colorbar()
        cbar.set_label("value")
    plt.savefig(save_dir+"network-matrix.png")

    plt.show()

def main():

    print("Training with inverse matrices: " + str(train_inverse))
    print("Training with log matrices:     " + str(train_log))

    batch_size = 30
    lr = 0.2
    num_epochs = 30

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    net = Network_Full(6, 100*100)
    #net = CovNet.Network_ReverseVGG(6)

    #net.apply(init_normal)
    net.apply(xavier)

    # use MSE loss function
    Loss = nn.L1Loss()

    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Training-Set/"
    save_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Plots/"
    train_data = MatrixDataset(training_dir, N_train, 0, train_log, train_inverse)
    valid_data = MatrixDataset(training_dir, N_valid, N_train, train_log, train_inverse)
    # test_data = MatrixDataset(training_dir, int(N*0.1), int(N*0.9))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    #train_in, train_out, test_in, test_out = read_training_set(training_dir, 0.8)

    # Train the network!
    t1 = time.time()
    net, train_loss, valid_loss = train(net, num_epochs, N_train, N_valid, batch_size, Loss, optimizer, train_loader, valid_loader)
    t2 = time.time()
    print("Done training!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    # Save the network to disk
    torch.save(net.state_dict(), 'network-VGG.params')

    plot_loss(train_loss, valid_loss, num_epochs, net, save_dir)

if __name__ == "__main__":
    main()