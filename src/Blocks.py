import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from src.Dataset import try_gpu, symmetric_exp, rearange_to_half, rearange_to_full
import src.Emulator as Emulator

#torch.set_default_dtype(torch.float64)

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
        # (batch size, length of each "sequence", "number of sequences")
        #X = X.reshape(X.shape[0], self.sequence_length, self.hidden_dim)

        X = self.addnorm1(X, self.attention(X, X, X))
        Y = F.leaky_relu(self.h1(X))
        Y = self.h2(X)
        X = self.addnorm2(X, Y)

        # reshape back to the shape of the input
        #X = X.reshape(X.shape[0], self.in_dim)
        return X

