import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time, math
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Total number of matrices in the training + test set
N = 1000

# wether to train using the percision matrix instead
train_inverse = False
# wether to train using the log of the matrix
train_log = True

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
        self.h7 = nn.Linear(5000, 5000)  # Hidden layer
        self.h8 = nn.Linear(5000, D_out)
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
        w = F.relu(self.h8(w))
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

        self.C1_1 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C1_2 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C1_3 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)

        self.C2_1 = nn.ConvTranspose2d(1, 1, kernel_size=10, stride=2) # 28x28
        self.C2_2 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C2_3 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C2_4 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)

        self.C3_1 = nn.ConvTranspose2d(1, 1, kernel_size=10) #37x27
        self.C3_2 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C3_3 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C3_4 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)

        self.C4_1 = nn.ConvTranspose2d(1, 1, kernel_size=10) # 46x46
        self.C4_2 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.C4_3 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)

        self.C5_1 = nn.ConvTranspose2d(1, 1, kernel_size=10, stride=2) #106x106
        self.C5_2 = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1)
        self.out = nn.ConvTranspose2d(1, 1, kernel_size=3, padding=1) # output layer

    # Define the forward propagation of the model
    def forward(self, X):
        # Work thru the fully connected part first
        w = F.relu(self.f3(F.relu(self.f2(F.relu(self.f1(X))))))
        w = w.view(-1, 1, 10, 10) # reshape into an "image"

        w = F.relu(self.C1_3(F.relu(self.C1_2(F.relu(self.C1_1(w))))))
        w = F.relu(self.C2_4(F.relu(self.C2_3(F.relu(self.C2_2(F.relu(self.C2_1(w))))))))
        w = F.relu(self.C3_4(F.relu(self.C3_3(F.relu(self.C3_2(F.relu(self.C3_1(w))))))))
        w = F.relu(self.C4_3(F.relu(self.C4_2(F.relu(self.C4_1(w))))))
        w = F.relu(self.out(F.relu(self.C5_2(F.relu(self.C5_1(w))))))
        return w


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

            if train_log == True:
                self.matrices[i] = symmetric_log(self.matrices[i])


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.params[idx], self.matrices[idx]

# Standard normal distribution
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

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

    batch_size = 25
    lr = 0.2
    num_epochs = 30

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    #net = Network_Full(6, 100*100)
    net = Network_ReverseVGG(6)

    #net.apply(init_normal)
    net.apply(xavier)

    # use MSE loss function
    Loss = nn.L1Loss()

    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Training-Set-Old/"
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
    net, train_loss, valid_loss = train(net, num_epochs, N_train, N_valid, batch_size, Loss, optimizer, train_loader, valid_loader)
    t2 = time.time()
    print("Done training!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    # Save the network to disk
    torch.save(net.state_dict(), 'network-VGG.params')

    plot_loss(train_loss, valid_loss, num_epochs, net, save_dir)

if __name__ == "__main__":
    main()