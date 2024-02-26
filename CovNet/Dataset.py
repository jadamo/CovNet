import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os, yaml
from easydict import EasyDict

torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------------------
# Dataset class to handle loading and pre-processing data
# ---------------------------------------------------------------------------
class MatrixDataset(torch.utils.data.Dataset):
    """
    Class that defines the set of matrices used by the covariance emulator.
    Builds on the torch.utils.data.Dataset primitive class
    """

    def __init__(self, data_dir:str, type:str, frac=1., train_gaussian_only=False,
                 pos_norm=0., neg_norm=0., use_gpu=True):
        """Initialize and load dataset of covariance matrices

        Args:
            data_dir: File location of training set
            type: Type of dataset to load. Options are ["training", "validataion", "testing"]
            frac: What fraction of the full dataset to use. Default 1.0
            train_gaussian_only: Whether to store only the gaussian term of the \
                covariance matrix. Default False
            pos_norm: The normalization value to be applied to positive elements \
                of each matrix. If 0, calculates this value from the input data. Default 0
            neg_norm: The normalization value to be applied to negative elements \
                of each matrix. If 0, calculates this value from the input data. Default 0
            use_gpu: Whether to attempt placing the data on the GPU. Default True
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
        data.close()
        self.N = self.matrices.shape[1]
        
        if frac != 1.:
            N_frac = int(len(self.matrices) * frac)
            self.matrices = self.matrices[0:N_frac]
            self.params = self.params[0:N_frac]

        self.pre_process()
        
        if use_gpu == True:
            self.params = self.params.to(try_gpu())
            self.matrices = self.matrices.to(try_gpu())

    def __len__(self):
        """Required function that returns the size of the dataset"""
        return self.matrices.shape[0]

    def __getitem__(self, idx):
        """Required function that returns a sample of the dataset given an index
        
        Args:
            idx: The sample indeces to return
        Returns:
            params: The input cosmology parameters associated with idx
            matrices: The covariance matrices associated with idx
        """
        return self.params[idx], self.matrices[idx]

    # def do_PCA(self, num_components, pca_dir="./"):
    #     """
    #     Converts the dataset to its principle components by either
    #     - fitting the dataset if no previous fit exists
    #     - loading a fit from a pickle file and using that
    #     NOTE: This is deprecated!
    #     @param num_components {int} the number of principle components to keep OR the desired accuracy
    #     @param pca_file {string} the location of pickle file with previous pca fit
    #     """
    #     flattened_data = rearange_to_half(self.matrices, 50).view(self.matrices.shape[0], 51*25)
    #     #flattened_data = (flattened_data + 1.) / 2

    #     self.pca = PCA(num_components)
    #     if not os.path.exists(pca_dir+"pca.pkl"):
    #         print("generating pca fit...")
    #         self.components = self.pca.fit_transform(flattened_data.cpu())
    #         print("Done, explained variance is {:0.4f}".format(np.cumsum(self.pca.explained_variance_ratio_)[-1]))

    #         self.components = torch.from_numpy(self.components).to(try_gpu())
    #         min_values = torch.min(self.components).detach()
    #         max_values = torch.max(self.components).detach()
    #         self.components = (self.components - min_values) / (max_values - min_values)

    #         with open(pca_dir+"pca.pkl", "wb") as pickle_file:
    #             pkl.dump([self.pca, min_values.to("cpu"), max_values.to("cpu")], pickle_file)
    #     else:
    #         with open(pca_dir+"pca.pkl", "rb") as pickle_file:
    #             load_data = pkl.load(pickle_file)
    #             self.pca = load_data[0]
    #             min_values = load_data[1].to(try_gpu())
    #             max_values = load_data[2].to(try_gpu())

    #         self.components = self.pca.transform(flattened_data.cpu())
    #         self.components = torch.from_numpy(self.components).to(try_gpu())
    #         self.components = (self.components - min_values) / (max_values - min_values)

    #     self.has_components = True
    #     return min_values, max_values

    def pre_process(self):
        """pre-processes the data to facilitate better training by:

        1: taking the Cholesky decomposition
        2: Taking the symmetric log
        3: Normalizing each element based on the sign
        """
        self.matrices = torch.linalg.cholesky(self.matrices)
        
        if self.norm_pos == 0:
            self.norm_pos = torch.log10(torch.max(self.matrices) + 1.)
        if self.norm_neg == 0:
            self.norm_neg = torch.log10(-1*torch.min(self.matrices) + 1.)

        # compress matrices first to save memory
        self.matrices = rearange_to_half(self.matrices)
        self.matrices = symmetric_log(self.matrices, self.norm_pos, self.norm_neg)
        self.matrices = rearange_to_full(self.matrices, self.cholesky)

    def get_full_matrix(self, idx:int):
        """reverses all data pre-processing to return the full covariance matrix
        
        Args:
            idx The index corresponding to the desired matrix
        Returns:
            mat (np array) The covariance matrix at index i as a numpy array
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
def rearange_to_half(C):
    """Compresses batch of lower-triangular matrices

    Takes a batch of matrices (B, N, N) and rearanges the lower half of each matrix
    to a rectangular (B, N+1, N/2) shape.

    Args:
        C: (3D Tensor) batch of square, lower triangular matrix to reshape
    Returns:
        L1 + L2: (3D Tensor) Batch of compressed matrices with zero elements removed
    """
    device = C.device
    B = C.shape[0]
    N = C.shape[1]
    N_half = int(N/2)
    L1 = torch.tril(C)[:,:,:N_half]; L2 = torch.tril(C)[:,:,N_half:]
    L1 = torch.cat((torch.zeros((B,1, N_half), device=device), L1), 1)
    L2 = torch.cat((torch.flip(L2, [1,2]), torch.zeros((B,1, N_half), device=device)),1)

    return L1 + L2

def rearange_to_full(C_half, lower_triangular=False):
    """Un-Compresses batch of lower-triangular matrices
    
    Takes a batch of half matrices (B, N+1, N/2) and reverses the rearangment to return full,
    symmetric matrices (B, N, N). This is the reverse operation of rearange_to_half()
    
    Args:
        C_half: (3D Tensor) Batch of compressed matrices with zeros removed
        lower_triangular: Whether or not matrices are lower triangular. If True, \ 
        reflects over the diagonal after re-shaping. Detault False
    """
    device = C_half.device
    N = C_half.shape[1] - 1
    N_half = int(N/2)
    B = C_half.shape[0]
    C_full = torch.zeros((B, N,N), device=device)
    C_full[:,:,:N_half] = C_full[:,:,:N_half] + C_half[:,1:,:]
    C_full[:,:,N_half:] = C_full[:,:,N_half:] + torch.flip(C_half[:,:-1,:], [1,2])
    L = torch.tril(C_full)
    U = torch.transpose(torch.tril(C_full, diagonal=-1),1,2)
    if lower_triangular: # <- if true we don't need to reflect over the diagonal, so just return L
        return L
    else:
        return L + U

# def reverse_pca(components, pca, min_values, max_values):
#     """
#     NOTE: obselete function!
#     """
#     components = (components * (max_values - min_values)) + min_values
#     matrix = torch.from_numpy(pca.inverse_transform(components.cpu().detach())).to(try_gpu()).view(-1, 51, 25)
#     matrix = rearange_to_full(matrix, 50, True)
#     return matrix

def symmetric_log(m, pos_norm:float, neg_norm:float):
    """Takes a a matrix and returns the normalized piece-wise logarithm 
    
    This function is used for pre-processing and uses the following equation:\n
    sym_log(x) =  log10(x+1) / pos_norm,  x >= 0\n
    sym_log(x) = -log10(-x+1) / neg_norm, x < 0\n

    Args:
        m: (3D Tensor) Batch of matrices to normalize
        pos_norm: Value to normalize positive elements with
        neg_norm: Value to normalize negative elements with
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

def symmetric_exp(m, pos_norm:float, neg_norm:float):
    """Takes a matrix and returns the piece-wise exponent

    sym_exp(x) =  10^( x*pos_norm) - 1,   x > 0\n
    sym_exp(x) = -10^-(x*neg_norm) + 1, x < 0\n
    This is the reverse operation of symmetric_log

    Args:
        m (3D Tensor) Batch of matrices to reverse-normalize
        pos_norm: Value used to normalize positive matrix elements
        neg_norm: Value used to normalize negative matrix elements
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

def load_config_file(config_file:str):
    """loads in the emulator config file as a dictionary object
    
    Args:
        config_file: Config file path and name to laod
    """
    with open(config_file, "r") as stream:
        try:
            config_dict = EasyDict(yaml.safe_load(stream))
        except:
            print("ERROR! Couldn't read yaml file")
            return None
        
    # some basic checks that your config file has the correct formating    
    if len(config_dict.mlp_dims) != config_dict.num_mlp_blocks + 1:
        print("ERROR! mlp dimensions not formatted correctly!")
        return None
    if len(config_dict.parameter_bounds) != config_dict.input_dim:
        print("ERROR! parameter bounds not formatted correctly!")
        return None
    
    return config_dict

def get_avg_loss(net, data):
    """Runs through the given set of matrices and returns the average loss value.
    
    Args:
        net: (Emulator object) The network to test
        data: (MatrixDataset object) the dataset to generate loss values for
    """
    avg_loss = 0
    net.eval()
    net = net.to("cpu")

    loader = torch.utils.data.DataLoader(data, batch_size=net.config_dict.batch_size, shuffle=True)
    for params, matrix in loader:
        prediction = net(params)
        avg_loss += F.l1_loss(prediction, matrix, reduction="sum").item()

    avg_loss /= len(data)
    net = net.to(try_gpu())
    return avg_loss

def organize_training_set(training_dir:str, train_frac:float, valid_frac:float, test_frac:float, 
                          params_dim:int, mat_dim:int, remove_old_files=True):
    """Takes a set of matrices and reorganizes them into training, validation, and tests sets
    
    Args:
        training_dir: Directory contaitning matrices to organize
        train_frac: Fraction of dataset to partition as the training set
        valid_frac: Fraction of dataset to partition as the validation set
        test_frac: Fraction of dataset to partition as the test set
        param_dim: Dimension of input parameter arrays
        mat_dim: Dimention of matrices
        remove_old_files: If True, deletes old data files after loading data into \
            memory and before re-organizing. Default True.
    """
    all_filenames = next(os.walk(training_dir), (None, None, []))[2]  # [] if no file

    all_params = np.array([], dtype=np.int64).reshape(0,params_dim)
    all_C_G = np.array([], dtype=np.int64).reshape(0,mat_dim, mat_dim)
    all_C_NG = np.array([], dtype=np.int64).reshape(0,mat_dim, mat_dim)

    # load in all the matrices internally (NOTE: memory intensive!)    
    for file in all_filenames:
        if "CovA-" in file:

            F_1 = np.load(training_dir+file, allow_pickle=True)

            params = F_1["params"]
            C_G = F_1["C_G"]
            C_NG = F_1["C_NG"]
            del F_1

            all_params = np.vstack([all_params, params])
            all_C_G = np.vstack([all_C_G, C_G])
            all_C_NG = np.vstack([all_C_NG, C_NG])

    # TODO: Add additional check for positive-definete-ness

    N = all_params.shape[0]
    N_train = int(N * train_frac)
    N_valid = int(N * valid_frac)
    N_test = int(N * test_frac)
    assert N_train + N_valid + N_test <= N

    valid_start = N_train
    valid_end = N_train + N_valid
    test_end = N_train + N_valid + N_test
    assert test_end - valid_end == N_test
    assert valid_end - valid_start == N_valid

    if remove_old_files == True:
        for file in all_filenames:
            if "CovA-" in file:
                os.remove(training_dir+file)

    print("splitting dataset into chunks of size [{:0.0f}, {:0.0f}, {:0.0f}]...".format(N_train, N_valid, N_test))

    np.savez(training_dir+"CovA-training.npz", 
                params=all_params[0:N_train], C_G=all_C_G[0:N_train], 
                C_NG=all_C_NG[0:N_train])
    np.savez(training_dir+"CovA-validation.npz", 
                params=all_params[valid_start:valid_end], 
                C_G=all_C_G[valid_start:valid_end], C_NG=all_C_NG[valid_start:valid_end])
    np.savez(training_dir+"CovA-testing.npz", 
                params=all_params[valid_end:test_end], 
                C_G=all_C_G[valid_end:test_end], C_NG=all_C_NG[valid_end:test_end])    


def try_gpu():
    """Return cuda or mps device if exists, otherwise return cpu"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device('cpu')
