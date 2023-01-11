import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#network to go from parameters to features
class Network_Features(nn.Module):
    def __init__(self, num_params, num_features):
        super().__init__()
        self.h1 = nn.Linear(num_params, 7)  # Hidden layer
        self.h2 = nn.Linear(7, 8)
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
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(51*25, 1000)
        #self.bn1 = nn.BatchNorm1d(2500)
        self.h2 = nn.Linear(1000, 1000)
        self.h3 = nn.Linear(1000, 750)
        self.h4 = nn.Linear(750, 500)
        self.h5 = nn.Linear(500, 100)
        self.h6 = nn.Linear(100, 100)
        self.h7 = nn.Linear(100, 50)
        self.bn2 = nn.BatchNorm1d(50)
        # self.h1 = nn.Linear(101*50, 1000)
        # self.h2 = nn.Linear(1000, 1000)
        # self.h3 = nn.Linear(1000, 1000)
        # self.h4 = nn.Linear(1000, 1000)
        # self.h5 = nn.Linear(1000, 1000)
        # self.h6 = nn.Linear(1000, 1000)
        #self.bn = nn.BatchNorm1d(1000)
        # 2 seperate layers - one for mu and one for log_var
        self.fmu = nn.Linear(50, 10)
        self.fvar = nn.Linear(50, 10)

    def reparameterize(self, mu, log_var):
        #if self.training:
        std = torch.exp(0.5*log_var)
        eps = std.data.new(std.size()).normal_()
        return mu + (eps * std) # sampling as if coming from the input space
        #else:
        #    return mu

    def forward(self, X):
        X = rearange_to_half(X, 50)
        X = X.view(-1, 51*25)

        X = F.leaky_relu(self.h1(X))
        #X = F.leaky_relu(self.h2(self.bn1(X)))
        X = F.leaky_relu(self.h2(X))
        X = F.leaky_relu(self.h3(X))
        X = F.leaky_relu(self.h4(X))
        X = F.leaky_relu(self.h5(X))
        X = F.leaky_relu(self.h6(X))
        X = F.leaky_relu(self.bn2(self.h7(X)))

        # using sigmoid here to keep log_var between 0 and 1
        mu = F.relu(self.fmu(X))
        log_var = F.relu(self.fvar(X))

        # The encoder outputs a distribution, so we need to draw some random sample from that
        # distribution in order to go through the decoder
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class Block_Decoder(nn.Module):
 
    def __init__(self, train_cholesky=False):
        super().__init__()
        self.train_cholesky = train_cholesky

        self.h1 = nn.Linear(10, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.h2 = nn.Linear(50, 100)
        self.h3 = nn.Linear(100, 100)
        self.h4 = nn.Linear(100, 500)
        self.h5 = nn.Linear(500, 750)
        self.h6 = nn.Linear(750, 1000)
        self.h7 = nn.Linear(1000, 1000)
        #self.bn2 = nn.BatchNorm1d(2500)
        self.out = nn.Linear(1000, 51*25)

    def forward(self, X):

        X = F.leaky_relu(self.bn1(self.h1(X)))
        X = F.leaky_relu(self.h2(X))
        X = F.leaky_relu(self.h3(X))
        X = F.leaky_relu(self.h4(X))
        X = F.leaky_relu(self.h5(X))
        X = F.leaky_relu(self.h6(X))
        X = F.leaky_relu(self.h7(X))
        #X = self.out(self.bn2(X))
        X = self.out(X)

        X = X.view(-1, 51, 25)
        X = rearange_to_full(X, 50, self.train_cholesky)
        return X

class Network_VAE(nn.Module):
    def __init__(self, train_cholesky=False):
        super().__init__()
        self.Encoder = Block_Encoder()
        self.Decoder = Block_Decoder(train_cholesky)

    def forward(self, X):
        # run through the encoder
        # assumes that z has been reparamaterized in the forward pass
        z, mu, log_var = self.Encoder(X)
        assert not True in torch.isnan(z)

        # run through the decoder
        X = self.Decoder(z)
        assert not True in torch.isnan(X)
        return X, mu, log_var


# class Block_Encoder_Quad(nn.Module):
#     """
#     Encoder block for a Variational Autoencoder (VAE). Input is an analytical covariance matrix, 
#     and the output is the normal distribution of a latent feature space
#     """
#     def __init__(self, N, do_half):
#         super().__init__()

#         self.do_half = do_half
#         in_dim = int(N*N) if not do_half else int((N+1)*N/2)
#         self.h1 = nn.Linear(in_dim, 750)
#         self.h2 = nn.Linear(750, 200)
#         self.h3 = nn.Linear(200, 50)
#         self.h4 = nn.Linear(50, 50)
#         self.bn = nn.BatchNorm1d(50)
#         # 2 seperate layers - one for mu and one for log_var
#         self.fmu = nn.Linear(50, 10)
#         self.fvar = nn.Linear(50, 10)

#     def reparameterize(self, mu, log_var):
#         #if self.training:
#         std = torch.exp(0.5*log_var)
#         eps = std.data.new(std.size()).normal_()
#         return mu + (eps * std) # sampling as if coming from the input space
#         #else:
#         #    return mu

#     def forward(self, X):
#         if self.do_half: 
#             X = rearange_to_half(X, 50)
#             X = X.view(-1, 51*25)
#         else:
#             X = X.reshape(-1, 50*50)

#         X = F.leaky_relu(self.h1(X))
#         X = F.leaky_relu(self.h2(X))
#         X = F.leaky_relu(self.h3(X))
#         X = F.leaky_relu(self.bn(self.h4(X)))

#         # using sigmoid here to keep log_var between 0 and 1
#         mu = F.relu(self.fmu(X))
#         log_var = F.relu(self.fvar(X))

#         # The encoder outputs a distribution, so we need to draw some random sample from that
#         # distribution in order to go through the decoder
#         z = self.reparameterize(mu, log_var)
#         return z, mu, log_var

# class Block_Decoder_Quad(nn.Module):
 
#     def __init__(self, N, do_half, train_cholesky=False):
#         super().__init__()
#         self.train_cholesky = train_cholesky
#         self.do_half = do_half
#         out_dim = int(N*N) if not do_half else int((N+1)*N/2)

#         self.h1 = nn.Linear(10, 50)
#         self.bn = nn.BatchNorm1d(50)
#         self.h2 = nn.Linear(50, 50)
#         self.h3 = nn.Linear(50, 200)
#         self.h4 = nn.Linear(200, 750)
#         self.out = nn.Linear(750, out_dim)

#     def forward(self, X):

#         X = F.leaky_relu(self.bn(self.h1(X)))
#         X = F.leaky_relu(self.h2(X))
#         X = F.leaky_relu(self.h3(X))
#         X = F.leaky_relu(self.h4(X))
#         X = self.out(X)

#         if self.do_half:
#             X = X.view(-1, 51, 25)
#             X = rearange_to_full(X, 50, self.train_cholesky)
#         else:
#             X = X.reshape(-1, 50, 50)
#         return X

# class Network_VAE_Quad(nn.Module):
#     def __init__(self, N, do_half, train_cholesky=False):
#         super().__init__()
#         self.Encoder = Block_Encoder_Quad(N, do_half)
#         self.Decoder = Block_Decoder_Quad(N, do_half, train_cholesky)

#     def forward(self, X):
#         # run through the encoder
#         # assumes that z has been reparamaterized in the forward pass
#         z, mu, log_var = self.Encoder(X)
#         assert not True in torch.isnan(z)

#         # run through the decoder
#         X = self.Decoder(z)
#         assert not True in torch.isnan(X)
#         return X, mu, log_var

# Dataset class to handle making training / validation / test sets
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset, train_log, train_gaussian=False, train_cholesky=False):
        self.params = torch.zeros([N, 7], device=try_gpu())
        self.matrices = torch.zeros([N, 50, 50], device=try_gpu())
        self.features = None
        self.offset = offset
        self.N = N

        self.has_features = False
        self.gaussian = train_gaussian
        self.cholesky = train_cholesky

        for i in range(N):

            # Load in the data from file
            idx = i + offset
            data = np.load(data_dir+"CovA-"+f'{idx:05d}'+".npz")
            self.params[i] = torch.from_numpy(data["params"])
            #self.params[i] = torch.from_numpy(data["params"])
            if self.cholesky and not self.gaussian:
                self.matrices[i] = torch.from_numpy(data["C_G"] + data["C_NG"])
            elif self.gaussian:
                self.matrices[i] = torch.from_numpy(data["C_G"])
            else:
                self.matrices[i] = torch.from_numpy(data["C_NG"])

            # if train_correlation:
            #     # the diagonal for correlation matrices is 1 everywhere, so let's store the diagonal there
            #     # instead to save space
            #     D = torch.sqrt(torch.diag(self.matrices[i]))
            #     D = torch.diag_embed(D)
            #     self.matrices[i] = torch.matmul(torch.linalg.inv(D), torch.matmul(self.matrices[i], torch.linalg.inv(D)))
            #     self.matrices[i] = self.matrices[i] + (symmetric_log(D) - torch.eye(100).to(try_gpu()))

            if train_cholesky:
                #self.matrices[i] = torch.linalg.inv(self.matrices[i])
                self.matrices[i] = torch.linalg.cholesky(self.matrices[i])

            if train_log == True:# and train_correlation == False:
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
    prediction = rearange_to_half(prediction, 50)
    target = rearange_to_half(target, 50)

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

def rearange_to_half(C, N):
    """
    Takes a batch of matrices (B, N, N) and rearanges the lower half of each matrix
    to a rectangular (B, N+1, N/2) shape.
    """
    N_half = int(N/2)
    B = C.shape[0]
    L1 = torch.tril(C)[:,:,:N_half]; L2 = torch.tril(C)[:,:,N_half:]
    L1 = torch.cat((torch.zeros((B,1, N_half), device=try_gpu()), L1), 1)
    L2 = torch.cat((torch.flip(L2, [1,2]), torch.zeros((B,1, N_half), device=try_gpu())),1)

    return L1 + L2

def rearange_to_full(C_half, N, train_cholesky=False):
    """
    Takes a batch of half matrices (B, N+1, N/2) and reverses the rearangment to return full,
    symmetric matrices (B, N, N). This is the reverse operation of rearange_to_half()
    """
    N_half = int(N/2)
    B = C_half.shape[0]
    C_full = torch.zeros((B, N,N), device=try_gpu())
    C_full[:,:,:N_half] = C_full[:,:,:N_half] + C_half[:,1:,:]
    C_full[:,:,N_half:] = C_full[:,:,N_half:] + torch.flip(C_half[:,:-1,:], [1,2])
    L = torch.tril(C_full)
    U = torch.transpose(torch.tril(C_full, diagonal=-1),1,2)
    if train_cholesky: # <- if true we don't need to reflect over the diagonal, so just return L
        return L
    else:
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

def predict_quad(decoder_q1, decoder_q2, decoder_q3, net_f1, net_f2, net_f3, params):
    """
    takes in the feature and VAE networks for each quadrant of the matrix
    and predicts the full covariance matrix
    """
    features = net_f1(params); C_q1 = decoder_q1(features.view(1,10))
    features = net_f2(params); C_q2 = decoder_q2(features.view(1,10))
    features = net_f3(params); C_q3 = decoder_q3(features.view(1,10))
    C = torch.zeros([C_q1.shape[0], 100, 100])
    C[:,:50,:50] = C_q1
    C[:,50:,50:] = C_q2
    C[:,:50,50:] = torch.transpose(C_q3, 1, 2)
    C[:,50:,:50] = C_q3
    return C

def corr_to_cov(C):
    """
    Takes the correlation matrix + variances and returns the full covariance matrix
    NOTE: This function assumes that the variances are stored in the diagonal of the correlation matrix
    """
    # Extract the log variances from the diagaonal
    D = torch.diag_embed(torch.diag(C)).to(try_gpu())
    C = C - D + torch.eye(100).to(try_gpu())
    D = symmetric_exp(D)
    C = torch.matmul(D, torch.matmul(C, D))
    return C

def try_gpu():
    """Return gpu() if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= 1:
        return torch.device('cuda')
    return torch.device('cpu')