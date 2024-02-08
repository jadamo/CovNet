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
    """
    Class that defines the set of matrices used by the covariance emulator.
    Builds on the torch.utils.data.Dataset primitive class
    """

    def __init__(self, data_dir, type, frac=1., train_gaussian_only=False,
                 pos_norm=0, neg_norm=0, use_gpu=True):
        """
        Initialize and load in dataset for training
        @param data_dir {string} location of training set
        @param type {string} type of data for the object. Options are ["training", "validataion", "testing"]
        @param frac {float} What fraction of the full dataset to use. Default 1
        @param train_gaussian_only {bool} whether to store only the gaussian term of the covariance matrix. Default False
        @param pos_norm {float} the normalization value to be applied to positive elements of each matrix. Default 0
        @param neg_norm {float} the normalization value to be applied to negative elements of each matrix. Default 0
        @param use_gpu {bool} Whether to attempt placing the data on the GPU. Default True
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
        """
        Required function that returns the size of the dataset
        @return N {int} the total number of matrixes in the dataset
        """
        return self.matrices.shape[0]

    def __getitem__(self, idx):
        """
        Required function that returns a sample of the dataset given an index
        @param idx {1D Tensor} the sample indeces to return
        @return params {2D Tensor} the input cosmology parameters associated with idx
        @return matrices {3D Tensor} the covariance matrices associated with idx
        @return idx {1D Tensor} the idx that was querried
        """
        return self.params[idx], self.matrices[idx], idx
        
    def get_quadrant(self, idx, quadrant):
        """
        NOTE: DEPRECATED FUNCTION
        """
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
        NOTE: This is deprecated!
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

        # Memory-intensive step
        self.matrices = rearange_to_half(self.matrices, self.N)
        self.matrices = symmetric_log(self.matrices, self.norm_pos, self.norm_neg)
        self.matrices = rearange_to_full(self.matrices, self.N, self.cholesky)

    def get_full_matrix(self, idx):
        """
        reverses all data pre-processing to return the full covariance matrix
        @param idx {int} the index corresponding to the desired matrix
        @return mat {np array} the non-processed covariance matrix as a numpy array
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
def Matrix_loss(prediction, target):
    """
    Calculates the aggregate L1 loss associated with the predicted and true covariance matrices
    @param prediction {3D Tensor} batch of matrices from the emulator
    @param target {3D Tensor} batch of matrices from the trianing set
    @return Loss {1D Tensor} L1 loss associated with the inputs
    """
    prediction = rearange_to_half(prediction, 50)
    target = rearange_to_half(target, 50)

    Loss = F.l1_loss(prediction, target, reduction="sum")

    return Loss

def rearange_to_half(C, N):
    """
    Takes a batch of matrices (B, N, N) and rearanges the lower half of each matrix
    to a rectangular (B, N+1, N/2) shape.
    @param C {3D Tensor} batch of square, lower triangular matrix to reshape
    @param N {int} dimension of matrices
    @return L1 + L2 {3D Tensor} batch of compressed matrices with zero elements removed
    """
    device = C.device
    N_half = int(N/2)
    B = C.shape[0]
    L1 = torch.tril(C)[:,:,:N_half]; L2 = torch.tril(C)[:,:,N_half:]
    L1 = torch.cat((torch.zeros((B,1, N_half), device=device), L1), 1)
    L2 = torch.cat((torch.flip(L2, [1,2]), torch.zeros((B,1, N_half), device=device)),1)

    return L1 + L2

def rearange_to_full(C_half, N, lower_triangular=False):
    """
    Takes a batch of half matrices (B, N+1, N/2) and reverses the rearangment to return full,
    symmetric matrices (B, N, N). This is the reverse operation of rearange_to_half()
    @param C_hale {3D Tensor} batch of compressed matrices with zeros removed
    @param N {int} dimension of matrices
    @param lower_triangular {bool} Whether or not matrices are lower triangular. Detault False
    @return C {3D Tensor} batch of uncompressed square matrices
    """
    device = C_half.device
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

def reverse_pca(components, pca, min_values, max_values):
    """
    NOTE: obselete function!
    """
    components = (components * (max_values - min_values)) + min_values
    matrix = torch.from_numpy(pca.inverse_transform(components.cpu().detach())).to(try_gpu()).view(-1, 51, 25)
    matrix = rearange_to_full(matrix, 50, True)
    return matrix

def combine_quadrants(C00, C22, C02):
    """
    NOTE: obselete function!
    """
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
    @param m {3D Tensor} batch of matrices to normalize
    @param pos_norm {float} value to normalize positive elements with
    @param neg_norm {float} value to normalize negative elements with
    @return m_norm {3D Tensor} batch of normalized matrices
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
    @param m {3D Tensor} batch of matrices to reverse-normalize
    @param pos_norm {float} value used to normalize positive matrix elements
    @param neg_norm {float} value used to normalize negative matrix elements
    @return m_true {3D Tensor} batch of matrices with their normalization reversed
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
    """
    loads in the emulator config file as a dictionary object
    @param config_file {string} config file path and name to laod
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

def organize_training_set(training_dir, train_frac, valid_frac, test_frac, 
                          params_dim, mat_dim, remove_old_files=True):
    """
    Takes a set of matrices and reorganizes them into training, validation, and tests sets
    @param training_dir {string} directory contaitning matrices to organize
    @param train_frac {float} fraction of dataset to partition as the training set
    @param valid_frac {float} fraction of dataset to partition as the validation set
    @param test_frac {float} fraction of dataset to partition as the test set
    @param param_dim {int} dimention of input parameter arrays
    @param mat_dim {int} dimention of matrices
    @param remove_old_files {bool} whether or not to delete old files before re-organizing
    """
    all_filenames = next(os.walk(training_dir), (None, None, []))[2]  # [] if no file

    all_params = np.array([], dtype=np.int64).reshape(0,params_dim)
    all_C_G = np.array([], dtype=np.int64).reshape(0,mat_dim, mat_dim)
    all_C_NG = np.array([], dtype=np.int64).reshape(0,mat_dim, mat_dim)

    # load in all the matrices internally (NOTE: memory intensive!)    
    for file in all_filenames:
        if "CovA-" in file:

            F_1 = np.load(training_dir+file)

            params = F_1["params"]
            C_G = F_1["C_G"]
            C_NG = F_1["C_NG"]
            del F_1

            all_params = np.vstack([all_params, params])
            all_C_G = np.vstack([all_C_G, C_G])
            all_C_NG = np.vstack([all_C_NG, C_NG])

    # TODO: Add additional check for positive-definete-ness
            
    if remove_old_files == True:
        for file in all_filenames:
            if "CovA-" in file:
                os.remove(training_dir+file)

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
    """
    Return cuda or mps device if exists, otherwise return cpu
    @return device {torch.device} either cpu, cuda, or mps device depending on machine compatability
    """
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device('cpu')
