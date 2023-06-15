import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time

#torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Emulator API Class - this is the class to call in a likelihood analysis
# ---------------------------------------------------------------------------
class CovNet():

    def __init__(self, net_dir, N, structure_flag=0,
                 pos_norm=5.91572, neg_norm=4.62748):
        """
        Initializes the covariance emulator in a warpper class by loading in the trained
        neural networks based on the specified options
        @param net_dir {string} location of trained networks
        @param N {int} the matrix dimensionality
        @param structure_flag {int} flag specifying the specific structure the network is (0 = fully connected ResNet, 1 = CNN ResNet)
        @param pos_norm {float} the normalization value to be applied to positive elements of each matrix
        @param neg_norm {float} the normalization value to be applied to negative elements of each matrix
        """
        self.structure_flag = structure_flag
        self.N = N

        self.norm_pos = pos_norm
        self.norm_neg = neg_norm

        self.net_VAE = Network_Emulator(structure_flag).to(try_gpu())
        self.net_VAE.eval()
        self.net_VAE.load_state_dict(torch.load(net_dir+'network-VAE.params', map_location=torch.device("cpu")))
        
        if structure_flag != 2:
            self.decoder = Block_Decoder(structure_flag).to(try_gpu())
            self.decoder.eval()
            self.decoder.load_state_dict(self.net_VAE.Decoder.state_dict())

            self.net_latent = Network_Latent().to(try_gpu())
            self.net_latent.load_state_dict(torch.load(net_dir+'network-latent.params', map_location=torch.device("cpu")))
            self.net_latent.load_state_dict(torch.load(net_dir+'network-latent.params', map_location=torch.device("cpu")))

    def get_covariance_matrix(self, params):
        """
        Uses the emulator to return a covariance matrix
        params -> secondary network -> decoder -> post-processing
        @param params {np array} the list of cosmology parameters to emulator a covariance matrix from
        @return C {np array} the emulated covariance matrix of size (N, N) where N was specified during initialization
        """
        params = torch.from_numpy(params).to(torch.float32)
        if self.structure_flag != 2:
            z = self.net_latent(params).view(1,6)
            matrix = self.decoder(z).view(1,self.N,self.N)
        else:
            matrix = self.net_VAE(params.view(1,6))

        matrix = symmetric_exp(matrix, self.norm_pos, self.norm_neg).view(self.N,self.N)
        matrix = torch.matmul(matrix, torch.t(matrix))

        return matrix.detach().numpy().astype(np.float64)

# ---------------------------------------------------------------------------
# Secondary (parameters -> latent space) network
# ---------------------------------------------------------------------------
class Network_Latent(nn.Module):
    def __init__(self, train_nuisance=False):
        super().__init__()
        self.h1 = nn.Linear(6, 6)  # Hidden layer
        self.h2 = nn.Linear(6, 6)
        self.h3 = nn.Linear(6, 6)
        self.h4 = nn.Linear(6, 6)

        self.h5 = nn.Linear(6, 6)
        self.h6 = nn.Linear(6, 6)
        self.h7 = nn.Linear(6, 6)
        self.h8 = nn.Linear(6, 6)

        self.h9 = nn.Linear(6, 6)
        self.h10 = nn.Linear(6, 6)
        self.h11 = nn.Linear(6, 6)
        self.h12 = nn.Linear(6, 6)

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
        

# ---------------------------------------------------------------------------
# VAE blocks
# ---------------------------------------------------------------------------
class Block_Full_ResNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.h1 = nn.Linear(dim_in, dim_out)
        self.bn1 = nn.BatchNorm1d(dim_out)
        self.h2 = nn.Linear(dim_out, dim_out)
        self.bn2 = nn.BatchNorm1d(dim_out)
        self.h3 = nn.Linear(dim_out, dim_out)
        self.bn3 = nn.BatchNorm1d(dim_out)
        self.h4 = nn.Linear(dim_out, dim_out)

        self.bn_skip = nn.BatchNorm1d(dim_out)
        self.skip = nn.Linear(dim_in, dim_out)

    def forward(self, X):
        residual = X
        X = F.leaky_relu(self.bn1(self.h1(X)))
        X = F.leaky_relu(self.bn2(self.h2(X)))
        X = F.leaky_relu(self.bn3(self.h3(X)))

        residual = self.bn_skip(self.skip(residual))
        X = F.leaky_relu(self.h4(X) + residual)
        return X

class Block_CNN_ResNet(nn.Module):
    def __init__(self, C_in, C_out, reverse=False):
        super().__init__()
        if reverse==False:
            self.c1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(C_out)
            self.c2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c3 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c4 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(C_out)
            self.c5 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c6 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c7 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(C_out)
            self.c8 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c9 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.c10 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(C_out)

            self.skip = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0)
            self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        else:
            self.c1 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(C_in)
            self.c2 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c3 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c4 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(C_in)
            self.c5 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c6 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c7 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(C_in)
            self.c8 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c9 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.c10 = nn.ConvTranspose2d(C_in, C_in, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(C_in)

            self.skip = nn.ConvTranspose2d(C_in, C_in, kernel_size=1, padding=0)
            self.pool = nn.ConvTranspose2d(C_in, C_out, kernel_size=2,stride=2)

    def forward(self, X):
        residual = X

        X = F.leaky_relu(self.bn1(self.c1(X)))
        X = F.leaky_relu(self.c2(X))
        X = F.leaky_relu(self.c3(X))
        X = F.leaky_relu(self.bn2(self.c4(X)))
        X = F.leaky_relu(self.c5(X))
        X = F.leaky_relu(self.c6(X))
        X = F.leaky_relu(self.bn3(self.c7(X)))
        X = F.leaky_relu(self.c8(X))
        X = F.leaky_relu(self.c9(X))
        X = self.bn4(self.c10(X))

        residual = self.skip(residual)
        X = F.leaky_relu(X + residual)
        X = self.pool(X)
        return X


# ---------------------------------------------------------------------------
# VAE network (Encoder + Decoder)
# ---------------------------------------------------------------------------
class Block_Encoder(nn.Module):
    """
    Encoder block for a Variational Autoencoder (VAE). Input is an analytical covariance matrix, 
    and the output is the normal distribution of a latent feature space
    """
    def __init__(self, structure_flag):
        super().__init__()
        self.structure_flag = structure_flag

        if self.structure_flag == 0 or self.structure_flag == 3:
            self.h1 = nn.Linear(51*25, 1000)
            self.resnet1 = Block_Full_ResNet(1000, 500)
            self.resnet2 = Block_Full_ResNet(500, 100)
            self.h2 = nn.Linear(100, 50)
            self.bn1 = nn.BatchNorm1d(50)
            self.h3 = nn.Linear(50, 25)

        elif self.structure_flag == 1:
            self.c1 = nn.Conv2d(1, 5, kernel_size=(4, 3), padding=1)
            self.resnet1 = Block_CNN_ResNet(5, 10) #(5,50,25) -> (5,25,12)
            self.resnet2 = Block_CNN_ResNet(10, 15) #(5,25,12) -> (7,12,6)
            self.resnet3 = Block_CNN_ResNet(15, 20) #(7,12,6)  -> (9,6,3)
            self.resnet4 = Block_CNN_ResNet(20, 30) #(9,6,3)   -> (9,3,1)

            self.f1 = nn.Linear(90, 50)
            self.f2 = nn.Linear(50, 25)

        # 2 seperate layers - one for mu and one for log_var
        self.fmu = nn.Linear(25, 6)
        if structure_flag != 3: self.fvar = nn.Linear(25, 6)

    def reparameterize(self, mu, log_var):
        #if self.training:
        std = torch.exp(0.5*log_var)
        eps = std.data.new(std.size()).normal_()
        return mu + (eps * std) # sampling as if coming from the input space
        #else:
        #    return mu

    def forward(self, X):
        X = rearange_to_half(X, 50)

        if self.structure_flag == 0 or self.structure_flag == 3:
            X = X.view(-1, 51*25)

            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = F.leaky_relu(self.bn1(self.h2(X)))
            X = F.leaky_relu(self.h3(X))

        elif self.structure_flag == 1:
            X = X.view(-1, 1, 51, 25)
            X = F.leaky_relu(self.c1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            X = torch.flatten(X, 1, 3)

            X = F.leaky_relu(self.f1(X))
            X = F.leaky_relu(self.f2(X))

        if self.structure_flag != 3:
            mu = self.fmu(X)
            log_var = self.fvar(X)

            # we're taking the exponent of this, so clamp to "reasonable" values to prevent overflow
            log_var = torch.clamp(log_var, min=-15, max=15)

            # The encoder outputs parameters of some distribution, so we need to draw some random sample from
            # that distribution in order to go through the decoder
            z = self.reparameterize(mu, log_var)
        else: 
            z = torch.tanh(self.fmu(X))
            mu = torch.zeros_like(z)
            log_var = torch.zeros_like(z)
        return z, mu, log_var

class Block_Decoder(nn.Module):
 
    def __init__(self, structure_flag):
        super().__init__()
        self.structure_flag = structure_flag

        if self.structure_flag == 0 or self.structure_flag == 3:
            self.h1 = nn.Linear(6, 25)
            self.h2 = nn.Linear(25, 50)
            self.h3 = nn.Linear(50, 100)
            self.resnet1 = Block_Full_ResNet(100, 500)
            self.resnet2 = Block_Full_ResNet(500, 1000)
            self.out = nn.Linear(1000, 51*25)

        elif self.structure_flag == 1:
            self.f1 = nn.Linear(6, 25)
            self.f2 = nn.Linear(25, 50)
            self.f3 = nn.Linear(50, 90)

            self.resnet1 = Block_CNN_ResNet(30, 20, True) #(20,3,1) -> (15,6,2)
            self.c1 = nn.ConvTranspose2d(20, 20, kernel_size=(1,2), padding=0, stride=1)
            self.resnet2 = Block_CNN_ResNet(20, 15, True)   #(4, 6, 3) -> (3, 12, 6)
            self.resnet3 = Block_CNN_ResNet(15, 10, True)   #(3, 12, 6) -> (2, 24, 12)
            self.resnet4 = Block_CNN_ResNet(10, 5, True)   #(2, 24, 12) -> (1, 48, 24)
            self.c2 = nn.ConvTranspose2d(5, 5, kernel_size=(4, 2))
            self.out = nn.ConvTranspose2d(5, 1, kernel_size=3, padding=1)

    def forward(self, X):

        if self.structure_flag == 0 or self.structure_flag == 3:
            X = F.leaky_relu(self.h1(X))
            X = F.leaky_relu(self.h2(X))
            X = F.leaky_relu(self.h3(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = torch.tanh(self.out(X))

        elif self.structure_flag == 1:
            X = F.leaky_relu(self.f1(X))
            X = F.leaky_relu(self.f2(X))
            X = F.leaky_relu(self.f3(X))
            X = X.reshape(-1, 30, 3, 1)
            X = self.resnet1(X)
            X = F.leaky_relu(self.c1(X))
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            X = F.leaky_relu(self.c2(X))
            X = self.out(X)

        X = X.view(-1, 51, 25)
        X = rearange_to_full(X, 50, True)
        return X

class Nulti_Headed_Attention(nn.Module):

    def __init__(self, hidden_dim, num_heads=2, dropout_prob=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layer_q = nn.Linear(hidden_dim, hidden_dim)
        self.layer_k = nn.Linear(hidden_dim, hidden_dim)
        self.layer_v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        #self.softmax = nn.Softmax(hidden_dim)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs, num_hiddens). 
        # Shape of output X: (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)

        X = X.permute(0, 2, 1, 3)

        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def dot_product_attention(self, q, k, v):
        dim = q.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        # calculate attention scores using the dot product
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dim)
        # normalize so that sum(scores) = 1 and all scores > 0
        #self.attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights = F.softmax(scores, dim=-1)
        # perform a batch matrix multiplaction to get the attention weights
        return torch.bmm(self.dropout(self.attention_weights), v)

    def forward(self, queries, keys, values):

        queries = self.transpose_qkv(self.layer_q(queries))
        keys    = self.transpose_qkv(self.layer_k(keys))
        values  = self.transpose_qkv(self.layer_v(values))

        X = self.dot_product_attention(queries, keys, values)
        X = self.out(self.transpose_output(X))
        return X

class Block_AddNorm(nn.Module):
    def __init__(self, shape, dropout_prob=0.):
        super().__init__()
        self.dropoiut = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(shape)
    def forward(self, X, Y):
        return self.layerNorm(self.dropoiut(Y) + X)

class Block_Transformer_Encoder(nn.Module):

    def __init__(self, in_dim, sequence_length, n_heads, dropout_prob=0.):
        super().__init__()
        self.in_dim = in_dim
        self.sequence_length = sequence_length
        self.hidden_dim = int(in_dim / sequence_length)

        self.attention = Nulti_Headed_Attention(self.hidden_dim, n_heads, dropout_prob).to(try_gpu())
        self.addnorm1 = Block_AddNorm(self.hidden_dim, dropout_prob)

        #feed-forward network
        self.h1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.h2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.addnorm2 = Block_AddNorm(self.hidden_dim, dropout_prob)

    def forward(self, X):
        # reshape input to
        # (batch size, previous hidden layer dimension)
        # (batch size, length of each "sequence", "Transformer hidden dimension")
        X = X.reshape(X.shape[0], self.sequence_length, self.hidden_dim)

        X = self.addnorm1(X, self.attention(X, X, X))
        Y = F.leaky_relu(self.h1(X))
        Y = self.h2(X)
        X = self.addnorm2(X, Y)

        # reshape back to the shape of the input
        X = X.reshape(X.shape[0], self.in_dim)
        return X

# class Transformer_Encoder(nn.Module):

#     def __init__(self, N=[51, 25], patch_size=[3,5]):
#         super().__init__()
#         self.patch_size = torch.Tensor(patch_size).int()
#         self.N = torch.Tensor(N).int()
#         self.n_patches = (self.N / self.patch_size).int().tolist()
#         self.patch_size = self.patch_size.tolist()

#         self.h1 = nn.Linear(6, 6)
#         #self.class_token = nn.Parameter(torch.rand(1, 10))

#         self.pos_embed = self.get_positional_embeddings(self.n_patches[0]*self.n_patches[1], 6)
#         self.pos_embed.requires_grad = False

#         # encoder blocks
#         self.encoder_block_1 = Block_Transformer(6, 10)
#         self.encoder_block_2 = Block_Transformer(6, 10)

#     def patchify(self, X):
#         patches = X.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
#         patches = patches.reshape(-1, self.n_patches[0]*self.n_patches[1], self.patch_size[0] * self.patch_size[1])

#         return patches

#     def get_positional_embeddings(self, sequence_length, d):
#         result = torch.ones(sequence_length, d)
#         for i in range(sequence_length):
#             for j in range(d):
#                 result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
#         return result

#     def forward(self, X):

#         #X = rearange_to_half(X, 50)
#         #print(X.shape)
#         # break image up into patches
#         #X = self.patchify(X)
#         X = X.view(-1, 1, 6)
#         X = X.repeat(1, 85, 1)
#         # linear embedding
#         tokens = self.h1(X)
#         # add classification token
#         #tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

#         # add positional information
#         pos_embed = self.pos_embed.repeat(tokens.shape[0], 1, 1)
#         X = tokens + pos_embed

#         X = self.encoder_block_1(X)
#         X = self.encoder_block_2(X)
#         #print(X.shape)
#         return X

class Network_Emulator(nn.Module):
    def __init__(self, structure_flag, dropout_prob=0.):
        super().__init__()
        self.structure_flag = structure_flag
        if structure_flag < 0 or structure_flag >= 5:
            print("ERROR! invalid value for structure flag! Currently can be between 0 and 4")
        elif structure_flag != 2 and structure_flag != 4:
            self.Encoder = Block_Encoder(structure_flag)
            self.Decoder = Block_Decoder(structure_flag)
        else:
            self.h1 = nn.Linear(6, 25)
            self.resnet1 = Block_Full_ResNet(25, 50)
            #if structure_flag == 4: self.transform1 = Block_Transformer(50, 2, 5, dropout_prob)
            self.resnet2 = Block_Full_ResNet(50, 100)
            #if structure_flag == 4: self.transform2 = Block_Transformer(100, 5, 5, dropout_prob)
            self.resnet3 = Block_Full_ResNet(100, 500)
            #if structure_flag == 4: self.transform1 = Block_Transformer_Encoder(500, 25, 5, dropout_prob)
            self.resnet4 = Block_Full_ResNet(500, 1000)
            if structure_flag == 4: self.transform1 = Block_Transformer_Encoder(1000, 25, 5, dropout_prob)
            self.out = nn.Linear(1000, 51*25)

        self.bounds = torch.tensor([[50, 100],
                                [0.01, 0.3],
                                [0.25, 1.65],
                                [1, 4],
                                [-4, 4],
                                [-4, 4]]).to(try_gpu())

    def normalize(self, params):

        params_norm = (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        return params_norm

    def forward(self, X):
        
        if self.structure_flag != 2 and self.structure_flag != 4:
            # run through the encoder
            # assumes that z has been reparamaterized in the forward pass
            z, mu, log_var = self.Encoder(X)
            assert not True in torch.isnan(z)

            # run through the decoder
            X = self.Decoder(z)
            X = torch.clamp(X, min=-12, max=12)
            assert not True in torch.isnan(X) 
            assert not True in torch.isinf(X)

            return X, mu, log_var
        else:
            X = self.normalize(X)
            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            #if self.structure_flag == 4: X = self.transform1(X)
            X = self.resnet2(X)
            #if self.structure_flag == 4: X = self.transform2(X)
            X = self.resnet3(X)
            #if self.structure_flag == 4: X = self.transform3(X)
            X = self.resnet4(X)
            if self.structure_flag == 4: X = self.transform1(X)
            X = torch.tanh(self.out(X))

            X = X.view(-1, 51, 25)
            X = rearange_to_full(X, 50, True)
            return X


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

        print(self.norm_pos, self.norm_neg)
        self.matrices = symmetric_log(self.matrices, self.norm_pos, self.norm_neg)
    
    def get_full_matrix(self, idx):
        """
        reverses all data pre-processing to return the full covariance matrix
        """

        pos_idx = torch.where(self.matrices[idx] >= 0)
        neg_idx = torch.where(self.matrices[idx] < 0)
        self.matrices[pos_idx] /= self.norm_pos
        self.matrices[neg_idx] /= self.norm_neg

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


# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------
def train_VAE(net, num_epochs, batch_size, beta, structure_flag,
              optimizer, train_loader, valid_loader, 
              print_progress = True, save_dir="", lr=0):
    """
    Train the VAE network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0
    net_save = Network_Emulator(structure_flag, True).to(try_gpu())

    train_loss = torch.zeros([num_epochs], device=try_gpu())
    valid_loss = torch.zeros([num_epochs], device=try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
        train_KLD_sub = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 50, 50))

            loss = VAE_loss(prediction, matrix, mu, log_var, beta)
            assert torch.isnan(loss) == False 
            assert torch.isinf(loss) == False

            train_loss_sub += loss.item()
            train_KLD_sub += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e8)    
            optimizer.step()

        # run through the validation set
        net.eval()
        valid_loss_sub = 0.
        valid_KLD_sub = 0.
        for (i, batch) in enumerate(valid_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 50, 50))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = VAE_loss(prediction, matrix, mu, log_var, beta)
            valid_loss_sub += loss.item()
            valid_KLD_sub  += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()

        # Aggregate loss information
        train_loss[epoch] = train_loss_sub / len(train_loader.dataset)
        valid_loss[epoch] = valid_loss_sub / len(valid_loader.dataset)
        if valid_KLD_sub < 1e-7 and beta != 0:
            print("WARNING! KLD term is close to 0, indicating potential posterior collapse!")

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for several epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss.dat")
                torch.save(valid_loss, save_dir+"valid_loss.dat")
                torch.save(net.state_dict(), save_dir+'network-VAE.params')
            worse_epochs = 0
        else:
            worse_epochs+=1

        if print_progress == True:
            print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, train_loss[epoch], valid_loss[epoch], worse_epochs))
            if beta != 0: print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(train_KLD_sub/len(train_loader.dataset), valid_KLD_sub/len(valid_loader.dataset)))

        if epoch > 15 and worse_epochs >= 15:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
            break
    print("initial lr {:0.5f}, bsize {:0.0f}: Best reconstruction validation loss was {:0.3f} after {:0.0f} epochs".format(lr, batch_size, best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss

def train_latent(net, num_epochs, optimizer, train_loader, valid_loader,
                 print_progress=True, save_dir=""):
    """
    Train the features network
    """
    best_loss = 1e10
    worse_epochs = 0
    net_save = Network_Latent()

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction = net(params)
            loss = features_loss(prediction, features)
            assert torch.isnan(loss) == False and torch.isinf(loss) == False

            avg_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # run through the validation set
        net.eval()
        avg_valid_loss = 0.
        for params, matrix, features in valid_loader:
            prediction = net(params)
            loss = features_loss(prediction, features)
            avg_valid_loss+= loss.item()

        # Aggregate loss information
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)

        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss-latent.dat")
                torch.save(valid_loss, save_dir+"valid_loss-latent.dat")
                torch.save(net.state_dict(), save_dir+'network-latent.params')
            worse_epochs = 0
        else:
            worse_epochs+= 1
        if epoch % 10 == 0 and print_progress == True:
            print("Epoch : {:d}, avg train loss: {:0.3f}\t best validation loss: {:0.3f}".format(epoch, avg_train_loss / len(train_loader.dataset), best_loss))
        if epoch > 30 and worse_epochs >= 20:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs. Stopping...")
            break
    print("Latent net: Best validation loss was {:0.4f} after {:0.0f} epochs".format(best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss

def train_MLP(net, num_epochs, batch_size, structure_flag,
              optimizer, train_loader, valid_loader,
              print_progress=True, save_dir="", lr=0):
    """
    Train the pure MLP network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0
    net_save = Network_Emulator(structure_flag, True).to(try_gpu())

    train_loss = torch.zeros([num_epochs], device=try_gpu())
    valid_loss = torch.zeros([num_epochs], device=try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params.view(batch_size, 6))

            loss = F.l1_loss(prediction, matrix, reduction="sum")
            assert torch.isnan(loss) == False 
            assert torch.isinf(loss) == False

            train_loss_sub += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e8)    
            optimizer.step()

        # run through the validation set
        net.eval()
        valid_loss_sub = 0.
        for (i, batch) in enumerate(valid_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params.view(batch_size, 6))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = F.l1_loss(prediction, matrix, reduction="sum")
            valid_loss_sub += loss.item()

        # Aggregate loss information
        train_loss[epoch] = train_loss_sub / len(train_loader.dataset)
        valid_loss[epoch] = valid_loss_sub / len(valid_loader.dataset)

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for several epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss.dat")
                torch.save(valid_loss, save_dir+"valid_loss.dat")
                torch.save(net.state_dict(), save_dir+'network-VAE.params')
            worse_epochs = 0
        else:
            worse_epochs+=1

        if print_progress == True: print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, train_loss[epoch], valid_loss[epoch], worse_epochs))

        if epoch > 15 and worse_epochs >= 15:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
            break
    print("lr {:0.5f}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(lr, batch_size, best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss