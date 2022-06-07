import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#network to go from parameters to features
class Network_Features(nn.Module):
    def __init__(self, num_params, num_features):
        super().__init__()
        self.h1 = nn.Linear(num_params, 6)  # Hidden layer
        self.h2 = nn.Linear(6, 8)
        self.h3 = nn.Linear(8, 10)  # Hidden layer
        self.out = nn.Linear(10, num_features)  # Output layer

    # Define the forward propagation of the model
    def forward(self, X):

        w = F.relu(self.h1(X))
        w = F.relu(self.h2(w))
        w = F.relu(self.h3(w))
        return self.out(w)

class Block_Encoder(nn.Module):
    """
    Encoder block for a Variational Autoencoder (VAE). Input is an analytical covariance matrix, 
    and the output is the normal distribution of a latent feature space
    """
    # def __init__(self):
    #     super().__init__()
    #     # TODO: find out if using channels and batch normalization would be good to include
    #     self.C1 = nn.Conv2d(1, 2, 4, stride=2, padding=1) # 50x50
    #     self.C2 = nn.Conv2d(2, 2, 3, padding=1) # 50x50
    #     self.C3 = nn.Conv2d(2, 4, 4, stride=2, padding=1) # 25x25
    #     self.C4 = nn.Conv2d(4, 4, 3) # 23x23
    #     self.C5 = nn.Conv2d(4, 6, 3, stride=2, padding=1) # 12x12
    #     self.C6 = nn.Conv2d(6, 6, 3) # 10x10

    #     self.f1 = nn.Linear(600, 250)
    #     self.f2 = nn.Linear(250, 80)
    #     # 2 seperate layers - one for mu and one for log_var
    #     self.fmu = nn.Linear(80, 20)
    #     self.fvar = nn.Linear(80, 20) # try smaller dimensionsl (10 - 20 is good maybe)

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(101*50, 2500)
        self.h2 = nn.Linear(2500, 1000)
        self.h3 = nn.Linear(1000, 500)
        self.h4 = nn.Linear(500, 100)
        self.h5 = nn.Linear(100, 50)
        self.bn = nn.BatchNorm1d(50)
        # 2 seperate layers - one for mu and one for log_var
        self.fmu = nn.Linear(50, 10)
        self.fvar = nn.Linear(50, 10)

    def reparameterize(self, mu, log_var):
        #if self.training:
        std = torch.exp(0.5*log_var)
        #eps = torch.randn_like(std)
        eps = std.data.new(std.size()).normal_()
        return mu + (eps * std) # sampling as if coming from the input space
        #else:
        #    return mu

    def forward(self, X):
        # X = F.leaky_relu(self.C1(X))
        # X = F.leaky_relu(self.C2(X))
        # X = F.leaky_relu(self.C3(X))
        # X = F.leaky_relu(self.C4(X))
        # X = F.leaky_relu(self.C5(X))
        # X = F.leaky_relu(self.C6(X))

        # X = X.view(-1, 1, 600)
        # X = F.leaky_relu(self.f1(X))
        # X = F.leaky_relu(self.f2(X))
        X = rearange_to_half(X)
        X = X.view(-1, 101*50)

        X = F.leaky_relu(self.h1(X))
        X = F.leaky_relu(self.h2(X))
        X = F.leaky_relu(self.h3(X))
        X = F.leaky_relu(self.h4(X))
        X = F.leaky_relu(self.bn(self.h5(X)))

        # using sigmoid here to keep log_var between 0 and 1
        mu = F.relu(self.fmu(X))
        log_var = F.relu(self.fvar(X))

        # The encoder outputs a distribution, so we need to draw some random sample from that
        # distribution in order to go through the decoder
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class Block_Decoder(nn.Module):
    #def __init__(self):
        # super().__init__()
        # self.f1 = nn.Linear(20, 80)  # Hidden layer
        # self.f2 = nn.Linear(80, 250)
        # self.f3 = nn.Linear(250, 600)

        # self.C1 = nn.ConvTranspose2d(6, 6, 3) #12x12
        # self.C2 = nn.ConvTranspose2d(6, 4, 3, stride=2, padding=1) #23x23
        # self.C3 = nn.ConvTranspose2d(4, 4, 3) #25x25
        # self.C4 = nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1) #49x49
        # self.C5 = nn.ConvTranspose2d(2, 2, 3, padding=1) #49x49
        # self.out = nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1) #100x100

    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(10, 50)
        self.bn = nn.BatchNorm1d(50)
        self.h2 = nn.Linear(50, 100)
        self.h3 = nn.Linear(100, 500)
        self.h4 = nn.Linear(500, 1000)
        self.h5 = nn.Linear(1000, 2500)
        self.out = nn.Linear(2500, 101*50)

    def forward(self, X):
        # X = F.leaky_relu(self.f1(X))
        # X = F.leaky_relu(self.f2(X))
        # X = F.leaky_relu(self.f3(X))

        # X = X.view(-1, 6, 10, 10)
        # X = F.leaky_relu(self.C1(X))
        # X = F.leaky_relu(self.C2(X))
        # X = F.leaky_relu(self.C3(X))
        # X = F.leaky_relu(self.C4(X))
        # X = F.leaky_relu(self.C5(X))
        # X = F.leaky_relu(self.out(X))
        # X = X.view(-1, 100, 100)

        X = F.leaky_relu(self.bn(self.h1(X)))
        X = F.leaky_relu(self.h2(X))
        X = F.leaky_relu(self.h3(X))
        X = F.leaky_relu(self.h4(X))
        X = F.leaky_relu(self.h5(X))
        X = self.out(X)

        # flip over the diagonal to ensure symmetry
        # L = torch.tril(X); U = torch.transpose(torch.tril(X, diagonal=-1),1,2)
        # X = L + U
        X = X.view(-1, 101, 50)
        X = rearange_to_full(X)
        return X

class Network_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = Block_Encoder()
        self.Decoder = Block_Decoder()

    def forward(self, X):
        # run through the encoder
        # assumes that z has been reparamaterized in the forward pass
        z, mu, log_var = self.Encoder(X)
        assert not True in torch.isnan(z)

        # run through the decoder
        X = self.Decoder(z)
        assert not True in torch.isnan(X)
        return X, mu, log_var

# Dataset class to handle making training / validation / test sets
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset, train_log, train_correlation=False, train_cholesky=False):
        self.params = torch.zeros([N, 6], device=try_gpu())
        self.matrices = torch.zeros([N, 100, 100], device=try_gpu())
        self.features = torch.zeros(N, 15, device=try_gpu())
        self.offset = offset
        self.N = N

        self.has_features = False
        self.correlation = train_correlation
        self.cholesky = train_cholesky

        for i in range(N):

            # Load in the data from file
            idx = i + offset
            data = np.load(data_dir+"CovA-"+f'{idx:05d}'+".npz")
            self.params[i] = torch.from_numpy(data["params"]) 
            self.matrices[i] = torch.from_numpy(data["C"])

            if train_correlation:
                # the diagonal for correlation matrices is 1 everywhere, so let's store the diagonal there
                # instead to save space
                D = torch.sqrt(torch.diag(self.matrices[i]))
                D = torch.diag_embed(D)
                self.matrices[i] = torch.matmul(torch.linalg.inv(D), torch.matmul(self.matrices[i], torch.linalg.inv(D)))
                self.matrices[i] = self.matrices[i] + (symmetric_log(D) - torch.eye(100))

            if train_cholesky:
                self.matrices[i] = torch.linalg.cholesky(self.matrices[i])

            if train_log == True and train_correlation == False:
                self.matrices[i] = symmetric_log(self.matrices[i])

    def add_features(self, z):
        self.features = z.detach()
        self.has_features = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.has_features:
            return self.params[idx], self.matrices[idx], self.features[idx]
        else:
            return self.params[idx], self.matrices[idx]

def VAE_loss(prediction, target, mu, log_var, beta=1.0):
    """
    Calculates the KL Divergence and reconstruction terms and returns the full loss function
    """
    RLoss = F.l1_loss(prediction, target, reduction="sum")
    #RLoss = torch.sqrt(((prediction - target)**2).sum())
    #RLoss = F.mse_loss(prediction, target, reduction="sum")
    KLD = 0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))
    #print(RLoss, KLD, torch.amin(prediction), torch.amax(prediction))
    return RLoss + (beta*KLD)

def features_loss(prediction, target):
    """
    Loss function for the parameters -> features network (currently MSE loss)
    """
    l1 = torch.pow(prediction - target, 2)
    return torch.mean(l1)

def rearange_to_half(C):
    """
    Takes a batch of matrices (B, 100, 100) and rearanges the lower half of each matrix
    to a rectangular (B, 101, 50) shape.
    """
    B = C.shape[0]
    L1 = torch.tril(C)[:,:,:50]; L2 = torch.tril(C)[:,:,50:]
    L1 = torch.cat((torch.zeros((B,1, 50), device=try_gpu()), L1), 1)
    L2 = torch.cat((torch.flip(L2, [1,2]), torch.zeros((B,1, 50), device=try_gpu())),1)

    return L1 + L2

def rearange_to_full(C_half):
    """
    Takes a batch of half matrices (B, 101, 50) and reverses the rearangment to return full,
    symmetric matrices (B, 100, 100). This is the reverse operation of rearange_to_half()
    """
    B = C_half.shape[0]
    C_full = torch.zeros((B, 100,100), device=try_gpu())
    C_full[:,:,:50] = C_full[:,:,:50] + C_half[:,1:,:]
    C_full[:,:,50:] = C_full[:,:,50:] + torch.flip(C_half[:,:-1,:], [1,2])
    L = torch.tril(C_full)
    U = torch.transpose(torch.tril(C_full, diagonal=-1),1,2)
    return L + U

def symmetric_log(m):
    """
    Takes a a matrix and returns a piece-wise logarithm for post-processing
    sym_log(x) =  log10(x+1),  x >= 0
    sym_log(x) = -log10(-x+1), x < 0
    """
    pos_m, neg_m = torch.zeros(m.shape, device=try_gpu()), torch.zeros(m.shape, device=try_gpu())
    pos_idx = torch.where(m >= 0)
    neg_idx = torch.where(m < 0)
    pos_m[pos_idx] = m[pos_idx]
    neg_m[neg_idx] = m[neg_idx]

    pos_m[pos_idx] = torch.log10(pos_m[pos_idx] + 1)
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -torch.log10(-1*neg_m[neg_idx] + 1)
    return pos_m + neg_m

def symmetric_exp(m):
    """
    Takes a matrix and returns the piece-wise exponent
    sym_exp(x) = 10^x - 1,   x > 0
    sym_exp(x) = -10^-x + 1, x < 0
    This is the reverse operation of symmetric_log
    """
    pos_m, neg_m = torch.zeros(m.shape, device=try_gpu()), torch.zeros(m.shape, device=try_gpu())
    pos_idx = torch.where(m >= 0)
    neg_idx = torch.where(m < 0)
    pos_m[pos_idx] = m[pos_idx]
    neg_m[neg_idx] = m[neg_idx]

    pos_m = 10**pos_m - 1
    pos_m[(pos_m == 1)] = 0
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -10**(-1*neg_m[neg_idx]) + 1
    return pos_m + neg_m

def corr_to_cov(C):
    """
    Takes the correlation matrix + variances and returns the full covariance matrix
    NOTE: This function assumes that the variances are stored in the diagonal of the correlation matrix
    """
    # Extract the log variances from the diagaonal
    D = torch.diag_embed(torch.diag(C))
    C = C - D + torch.eye(100)
    D = symmetric_exp(D)
    C = torch.matmul(D, torch.matmul(C, D))
    return C

def try_gpu():
    """Return gpu() if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')