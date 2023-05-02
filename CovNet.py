import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#torch.set_default_dtype(torch.float64)

class CovNet():

    def __init__(self, net_dir, N, train_cholesky=True, train_nuisance=False):

        self.train_cholesky = train_cholesky
        self.N = N

        self.net_VAE = Network_VAE(train_cholesky).to(try_gpu())
        self.decoder = Block_Decoder(train_cholesky).to(try_gpu())
        self.net_latent = Network_Latent().to(try_gpu())
        self.net_VAE.eval()
        self.decoder.eval()

        self.net_VAE.load_state_dict(torch.load(net_dir+'network-VAE.params'))
        self.decoder.load_state_dict(self.net_VAE.Decoder.state_dict())
        self.net_latent.load_state_dict(torch.load(net_dir+'network-latent.params'))

    def get_covariance_matrix(self, params):
        params = torch.from_numpy(params).to(torch.float32)
        z = self.net_latent(params).view(1,6)
        matrix = self.decoder(z).view(1,self.N,self.N)

        matrix = symmetric_exp(matrix).view(self.N,self.N)
        if self.train_cholesky == True:
            matrix = torch.matmul(matrix, torch.t(matrix))

        return matrix.detach().numpy().astype(np.float64)

#network to go from parameters to features
class Network_Latent(nn.Module):
    def __init__(self, train_nuisance=False):
        super().__init__()
        self.h1 = nn.Linear(6, 6)  # Hidden layer
        self.h2 = nn.Linear(6, 6)
        self.h3 = nn.Linear(6, 6)
        self.h4 = nn.Linear(6, 6)
        #self.skip1 = nn.Linear(6, 6)

        self.h5 = nn.Linear(6, 6)
        self.h6 = nn.Linear(6, 6)
        self.h7 = nn.Linear(6, 6)
        self.h8 = nn.Linear(6, 6)
        #self.skip2 = nn.Linear(6, 6)

        self.h9 = nn.Linear(6, 6)
        self.h10 = nn.Linear(6, 6)
        self.h11 = nn.Linear(6, 6)
        self.h12 = nn.Linear(6, 6)
        #self.skip3 = nn.Linear(6, 6)

        self.h13 = nn.Linear(6, 6)
        self.h14 = nn.Linear(6, 6)
        self.h15 = nn.Linear(6, 6)
        self.h16 = nn.Linear(6, 6)

        self.h17 = nn.Linear(6, 6)
        self.h18 = nn.Linear(6, 6)
        self.out = nn.Linear(6, 6)  # Output layer

        self.bounds = torch.tensor([[50, 100],
                                    [0.01, 0.3],
                                    [0.25, 1.65],
                                    [1, 4],
                                    [-4, 4],
                                    [-4, 4]])

    def normalize(self, params):

        params_norm = (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        return params_norm

    # Define the forward propagation of the model
    def forward(self, X):

        X = self.normalize(X)
        residual = X

        w = F.leaky_relu(self.h1(X))
        w = F.leaky_relu(self.h2(w))
        w = F.leaky_relu(self.h3(w))
        w = self.h4(w) + residual; residual = w
        w = F.leaky_relu(self.h5(w))
        w = F.leaky_relu(self.h6(w))
        w = F.leaky_relu(self.h7(w))
        w = self.h8(w) + residual; residual = w
        w = F.leaky_relu(self.h9(w))
        w = F.leaky_relu(self.h10(w))
        w = F.leaky_relu(self.h11(w))
        w = self.h12(w) + residual; residual = w
        w = F.leaky_relu(self.h13(w))
        w = F.leaky_relu(self.h14(w))
        w = F.leaky_relu(self.h15(w))
        w = self.h16(w) + residual; residual = w
        w = F.leaky_relu(self.h17(w))
        w = F.leaky_relu(self.h18(w))
        return self.out(w)

class Block_ResNet(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.h1 = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.h2 = nn.Linear(dim_out, dim_out)
        self.h3 = nn.Linear(dim_out, dim_out)
        self.h4 = nn.Linear(dim_out, dim_out)

        self.bn_skip = nn.BatchNorm1d(dim_out)
        self.skip = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        residual = X

        X = F.leaky_relu(self.h1(X))
        X = self.bn(X)
        X = F.leaky_relu(self.h2(X))
        X = F.leaky_relu(self.h3(X))
        residual = self.bn_skip(self.skip(residual))
        X = F.leaky_relu(self.h4(X) + residual)
        return X

class Block_Encoder(nn.Module):
    """
    Encoder block for a Variational Autoencoder (VAE). Input is an analytical covariance matrix, 
    and the output is the normal distribution of a latent feature space
    """
    def __init__(self):
        super().__init__()

        self.h1 = nn.Linear(51*25, 1000)
        self.resnet1 = Block_ResNet(1000, 500)
        self.resnet2 = Block_ResNet(500, 100)

        self.h2 = nn.Linear(100, 50)
        # 2 seperate layers - one for mu and one for log_var
        self.fmu = nn.Linear(50, 6)
        self.fvar = nn.Linear(50, 6)

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
        X = self.resnet1(X)
        X = self.resnet2(X)
        #X = self.resnet3(X)
        X = F.leaky_relu(self.h2(X))

        # using sigmoid here to keep log_var between 0 and 1
        mu = self.fmu(X)
        log_var = self.fvar(X)

        # The encoder outputs parameters of some distribution, so we need to draw some random sample from
        # that distribution in order to go through the decoder
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class Block_Decoder(nn.Module):
 
    def __init__(self, train_cholesky=False):
        super().__init__()
        self.train_cholesky = train_cholesky

        self.h1 = nn.Linear(6, 50)
        self.h2 = nn.Linear(50, 100)
        #self.resnet1 = Block_ResNet(50, 100)
        self.resnet1 = Block_ResNet(100, 500)
        self.resnet2 = Block_ResNet(500, 1000)
        self.out = nn.Linear(1000, 51*25)

    def forward(self, X):

        X = F.leaky_relu(self.h1(X))
        X = F.leaky_relu(self.h2(X))
        X = self.resnet1(X)
        X = self.resnet2(X)
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
        if True in torch.isnan(X) or True in torch.isinf(X):
            print("ERROR! nan or infinity found in decoder output! Fixing to some value...")
            X = torch.nan_to_num(X, posinf=100, neginf=-100)
        return X, mu, log_var

# Dataset class to handle making training / validation / test sets
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset, train_nuisance, train_cholesky=True, 
                 train_gaussian_only=False, train_T0_only=False):
        """
        Initialize and load in dataset for training
        @param data_dir {string} location of training set
        @param N {int} size of training set
        @param offset {int} index number to begin reading training set (used when splitting set into training / validation / test sets)
        @param train_nuisance {bool} whether you will be varying nuisance parameters when training
        @param train_cholesky {bool} whether to represent each matrix as its cholesky decomposition
        @param train_gaussian {bool} whether to store only the gaussian term of the covariance matrix (for testing)
        @param train_T0_only {bool} whether to store only the T0 term of the covariance matrix (for testing)
        """

        assert not (train_gaussian_only == True and train_T0_only == True)
        assert not (train_T0_only == True and train_nuisance == True)

        num_params=6 if train_nuisance==False else 10
        self.params = torch.zeros([N, num_params], device=try_gpu())
        self.matrices = torch.zeros([N, 50, 50], device=try_gpu())
        self.features = None
        self.offset = offset
        self.N = N

        self.has_latent_space = False

        self.cholesky = train_cholesky
        self.T0_only = train_T0_only
        self.gaussian_only = train_gaussian_only

        for i in range(N):

            # Load in the data from file
            idx = i + offset
            data = np.load(data_dir+"CovA-"+f'{idx:05d}'+".npz")
            self.params[i] = torch.from_numpy(data["params"])

            # store specific terms of each matrix depending on the circumstances
            if self.gaussian_only:
                self.matrices[i] = torch.from_numpy(data["C_G"])
            elif self.T0_only:
                self.matrices[i] = torch.from_numpy(data["C_T0"])
            else:
                self.matrices[i] = torch.from_numpy(data["C_G"] + data["C_SSC"] + data["C_T0"])

            # if train_correlation:
            #     # the diagonal for correlation matrices is 1 everywhere, so let's store the diagonal there
            #     # instead to save space
            #     D = torch.sqrt(torch.diag(self.matrices[i]))
            #     D = torch.diag_embed(D)
            #     self.matrices[i] = torch.matmul(torch.linalg.inv(D), torch.matmul(self.matrices[i], torch.linalg.inv(D)))
            #     self.matrices[i] = self.matrices[i] + (symmetric_log(D) - torch.eye(100).to(try_gpu()))

            if train_cholesky:
                self.matrices[i] = torch.linalg.cholesky(self.matrices[i])

            self.matrices[i] = symmetric_log(self.matrices[i])

    def add_latent_space(self, z):
        self.latent_space = z.detach()
        self.has_latent_space = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.has_latent_space:
            return self.params[idx], self.matrices[idx], self.latent_space[idx]
        else:
            return self.params[idx], self.matrices[idx]
        
    def get_full_matrix(self, idx):
        """
        reverses all data pre-processing to return the full covariance matrix
        """
        # reverse logarithm (always true)
        mat = symmetric_exp(self.matrices[idx])

        if self.cholesky == True:
            mat = torch.matmul(mat, torch.t(mat))

        return mat.detach().numpy()


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
    loss = F.l1_loss(prediction, target, reduction="sum")
    #loss = F.mse_loss(prediction, target, reduction="sum")

    return loss

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