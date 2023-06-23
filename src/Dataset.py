import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# ---------------------------------------------------------------------------
# Dataset class to handle loading and pre-processing data
# ---------------------------------------------------------------------------
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, N, offset, train_gaussian_only=False,
                 pos_norm=5.91572, neg_norm=4.62748):
        """
        Initialize and load in dataset for training
        @param data_dir {string} location of training set
        @param N {int} size of training set
        @param offset {int} index number to begin reading training set (used when splitting set into training / validation / test sets)
        @param train_gaussian_only {bool} whether to store only the gaussian term of the covariance matrix (for testing)
        @param pos_norm {float} the normalization value to be applied to positive elements of each matrix
        @param neg_norm {float} the normalization value to be applied to negative elements of each matrix
        """

        num_params=6
        self.params = torch.zeros([N, num_params])
        self.matrices = torch.zeros([N, 50, 50])
        self.features = None
        self.offset = offset
        self.N = N

        self.has_latent_space = False

        self.cholesky = True
        self.gaussian_only = train_gaussian_only

        self.norm_pos = pos_norm
        self.norm_neg = neg_norm

        for i in range(N):

            idx = i + offset
            data = np.load(data_dir+"CovA-"+f'{idx:05d}'+".npz")
            self.params[i] = torch.from_numpy(data["params"][:6])

            # store specific terms of each matrix depending on the circumstances
            if self.gaussian_only:
                self.matrices[i] = torch.from_numpy(data["C_G"])
            else:
                self.matrices[i] = torch.from_numpy(data["C_G"] + data["C_SSC"] + data["C_T0"])

        self.params = self.params.to(try_gpu())
        self.matrices = self.matrices.to(try_gpu())

        self.pre_process()

    def add_latent_space(self, z):
        # training latent net seems to be faster on cpu, so move data there
        self.latent_space = z.detach().to(torch.device("cpu"))
        self.params = self.params.to(torch.device("cpu"))
        self.has_latent_space = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.has_latent_space:
            return self.params[idx], self.matrices[idx], self.latent_space[idx]
        else:
            return self.params[idx], self.matrices[idx]
        
    def pre_process(self):
        """
        pre-processes the data to facilitate better training by
        1: taking the Cholesky decomposition
        2: Taking the symmetric log
        3. Normalizing each element based on the sign
        """
        self.matrices = torch.linalg.cholesky(self.matrices)

        if self.norm_pos == 0:
            self.norm_pos = torch.log10(torch.max(self.matrices) + 1.)
        if self.norm_neg == 0:
            self.norm_neg = torch.log10(-1*torch.min(self.matrices) + 1.)

        self.matrices = symmetric_log(self.matrices, self.norm_pos, self.norm_neg)
    
    def get_full_matrix(self, idx):
        """
        reverses all data pre-processing to return the full covariance matrix
        """

        # pos_idx = torch.where(self.matrices[idx] >= 0)
        # neg_idx = torch.where(self.matrices[idx] < 0)
        # self.matrices[pos_idx] *= self.norm_pos
        # self.matrices[neg_idx] *= self.norm_neg

        # reverse logarithm (always true)
        mat = symmetric_exp(self.matrices[idx], self.norm_pos, self.norm_neg)

        if self.cholesky == True:
            mat = torch.matmul(mat, torch.t(mat))

        return mat.detach().numpy()

# ---------------------------------------------------------------------------
# Other helper functions
# ---------------------------------------------------------------------------
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

def rearange_to_full(C_half, N, return_cholesky=False):
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
    if return_cholesky: # <- if true we don't need to reflect over the diagonal, so just return L
        return L
    else:
        return L + U

def symmetric_log(m, pos_norm, neg_norm):
    """
    Takes a a matrix and returns a normalized piece-wise logarithm for post-processing
    sym_log(x) =  log10(x+1) / pos_norm,  x >= 0
    sym_log(x) = -log10(-x+1) / neg_norm, x < 0
    """
    pos_m, neg_m = torch.zeros(m.shape, device=try_gpu()), torch.zeros(m.shape, device=try_gpu())
    pos_idx = torch.where(m >= 0)
    neg_idx = torch.where(m < 0)
    pos_m[pos_idx] = m[pos_idx]
    neg_m[neg_idx] = m[neg_idx]

    pos_m[pos_idx] = torch.log10(pos_m[pos_idx] + 1)
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -torch.log10(-1*neg_m[neg_idx] + 1)
    return (pos_m / pos_norm) + (neg_m / neg_norm)

def symmetric_exp(m, pos_norm, neg_norm):
    """
    Takes a matrix and returns the piece-wise exponent
    sym_exp(x) =  10^( x*pos_norm) - 1,   x > 0
    sym_exp(x) = -10^-(x*neg_norm) + 1, x < 0
    This is the reverse operation of symmetric_log
    """
    pos_m, neg_m = torch.zeros(m.shape, device=try_gpu()), torch.zeros(m.shape, device=try_gpu())
    pos_idx = torch.where(m >= 0)
    neg_idx = torch.where(m < 0)
    pos_m[pos_idx] = m[pos_idx] * pos_norm
    neg_m[neg_idx] = m[neg_idx] * neg_norm

    pos_m = 10**pos_m - 1
    pos_m[(pos_m == 1)] = 0
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -10**(-1*neg_m[neg_idx]) + 1

    return pos_m + neg_m

def try_gpu():
    """Return gpu() if exists, otherwise return cpu()."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')
