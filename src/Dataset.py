import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle as pkl
import os, yaml
from easydict import EasyDict

from sklearn.decomposition import PCA

torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------------------
# Dataset class to handle loading and pre-processing data
# ---------------------------------------------------------------------------
class MatrixDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, type, frac=1., train_gaussian_only=False,
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

        if type=="training":
            file = data_dir+"CovA-training.npz"
        elif type=="training-small":
            file = data_dir+"CovA-training-small.npz"
        elif type=="validation":
            file = data_dir+"CovA-validation.npz"
        elif type=="testing":
            file = data_dir+"CovA-testing.npz"
        else: print("ERROR! Invalid dataset type! Must be [training, validation, testing]")

        self.has_latent_space = False
        self.has_components = False

        self.cholesky = True
        self.gaussian_only = train_gaussian_only

        self.norm_pos = pos_norm
        self.norm_neg = neg_norm

        data = np.load(file)
        self.params = torch.from_numpy(data["params"]).to(torch.float32)

        # store specific terms of each matrix depending on the circumstances
        if self.gaussian_only:
            self.matrices = torch.from_numpy(data["C_G"]).to(torch.float32)
        else:
            self.matrices = torch.from_numpy(data["C_G"] + data["C_NG"]).to(torch.float32)

        if frac != 1.:
            N_frac = int(len(self.matrices) * frac)
            self.matrices = self.matrices[0:N_frac]
            self.params = self.params[0:N_frac]

        self.pre_process()

        self.params = self.params.to(try_gpu())
        self.matrices = self.matrices.to(try_gpu())

    def add_latent_space(self, z):
        # training latent net seems to be faster on cpu, so move data there
        self.latent_space = z.detach().to(torch.device("cpu"))
        self.params = self.params.to(torch.device("cpu"))
        self.has_latent_space = True

    def __len__(self):
        return self.matrices.shape[0]

    def __getitem__(self, idx):
        if self.has_latent_space:
            return self.params[idx], self.matrices[idx], self.latent_space[idx]
        elif self.has_components:
            return self.params[idx], self.matrices[idx], self.components[idx]
        else:
            return self.params[idx], self.matrices[idx], idx
        
    def get_quadrant(self, idx, quadrant):
        if quadrant == "00":
            return self.matrices[idx][:, :25, :25]
        elif quadrant=="22":
            return self.matrices[idx][:, 25:, 25:]
        elif quadrant=="02":
            return self.matrices[idx][:, 25:, :25]

    def do_PCA(self, num_components, pca_dir="./"):
        """
        Converts the dataset to its principle components by either
        - fitting the dataset if no previous fit exists
        - loading a fit from a pickle file and using that
        @param num_components {int} the number of principle components to keep OR the desired accuracy
        @param pca_file {string} the location of pickle file with previous pca fit
        """
        flattened_data = rearange_to_half(self.matrices, 50).view(self.matrices.shape[0], 51*25)
        #flattened_data = (flattened_data + 1.) / 2

        self.pca = PCA(num_components)
        if not os.path.exists(pca_dir+"pca.pkl"):
            print("generating pca fit...")
            self.components = self.pca.fit_transform(flattened_data.cpu())
            print("Done, explained variance is {:0.4f}".format(np.cumsum(self.pca.explained_variance_ratio_)[-1]))

            self.components = torch.from_numpy(self.components).to(try_gpu())
            min_values = torch.min(self.components).detach()
            max_values = torch.max(self.components).detach()
            self.components = (self.components - min_values) / (max_values - min_values)

            with open(pca_dir+"pca.pkl", "wb") as pickle_file:
                pkl.dump([self.pca, min_values.to("cpu"), max_values.to("cpu")], pickle_file)
        else:
            with open(pca_dir+"pca.pkl", "rb") as pickle_file:
                load_data = pkl.load(pickle_file)
                self.pca = load_data[0]
                min_values = load_data[1].to(try_gpu())
                max_values = load_data[2].to(try_gpu())

            self.components = self.pca.transform(flattened_data.cpu())
            self.components = torch.from_numpy(self.components).to(try_gpu())
            self.components = (self.components - min_values) / (max_values - min_values)

        self.has_components = True
        return min_values, max_values

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
        @ param idx {int} the index corresponding to the desired matrix
        @ return mat {np array} the non-processed covariance matrix as a numpy array
        """
        # reverse logarithm and normalization (always true)
        # converting to np array before matmul seems to be more numerically stable
        mat = symmetric_exp(self.matrices[idx], self.norm_pos, self.norm_neg).detach().numpy()

        if self.cholesky == True:
            mat = np.matmul(mat, mat.T)

        return mat

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

def reverse_pca(components, pca, min_values, max_values):
    components = (components * (max_values - min_values)) + min_values
    matrix = torch.from_numpy(pca.inverse_transform(components.cpu().detach())).to(try_gpu()).view(-1, 51, 25)
    matrix = rearange_to_full(matrix, 50, True)
    return matrix

def combine_quadrants(C00, C22, C02):

    patches = torch.vstack([C00.to(try_gpu()), torch.zeros(1, 25, 25).to(try_gpu()), C02.to(try_gpu()), C22.to(try_gpu())])
    patches = patches.reshape(-1, 2, 2, 25, 25)
    C_full = patches.permute(0, 1, 3, 2, 4).contiguous()
    C_full = C_full.view(-1, 50, 50)
    return C_full

def symmetric_log(m, pos_norm, neg_norm):
    """
    Takes a a matrix and returns a normalized piece-wise logarithm for post-processing
    sym_log(x) =  log10(x+1) / pos_norm,  x >= 0
    sym_log(x) = -log10(-x+1) / neg_norm, x < 0
    """
    device = m.device
    pos_m, neg_m = torch.zeros(m.shape, device=device), torch.zeros(m.shape, device=device)
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
    device = m.device
    pos_m, neg_m = torch.zeros(m.shape, device=device), torch.zeros(m.shape, device=device)
    pos_idx = torch.where(m >= 0)
    neg_idx = torch.where(m < 0)
    pos_m[pos_idx] = m[pos_idx] * pos_norm
    neg_m[neg_idx] = m[neg_idx] * neg_norm

    pos_m = 10**pos_m - 1
    pos_m[(pos_m == 1)] = 0
    # for negative numbers, treat log(x) = -log(-x)
    neg_m[neg_idx] = -10**(-1*neg_m[neg_idx]) + 1

    return pos_m + neg_m

def load_config_file(config_file):
    """loads in the emulator config file as a dictionary object"""
    with open(config_file, "r") as stream:
        try:
            config_dict = EasyDict(yaml.safe_load(stream))
        except:
            print("ERROR! Couldn't read yaml file")
            return
        
    # some basic checks that your config file has the correct formating    
    if len(config_dict.mlp_dims) != config_dict.num_mlp_blocks + 1:
        print("ERROR! mlp dimensions not formatted correctly!")
        return
    if len(config_dict.parameter_bounds) != config_dict.input_dim:
        print("ERROR! parameter bounds not formatted correctly!")
        return
    
    return config_dict

def try_gpu():
    """Return gpu() if exists, otherwise return cpu()."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    # elif torch.mps.is_available():
    #     return torch.device("mps")
    return torch.device('cpu')
