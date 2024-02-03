import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle as pkl
import yaml

import CovNet.Blocks as Blocks
import CovNet.Dataset as Dataset
from CovNet.Dataset import try_gpu

# ---------------------------------------------------------------------------
# Emulator API Class - this is the class to call in a likelihood analysis
# ---------------------------------------------------------------------------
class CovNet():

    def __init__(self, net_dir):
        """
        Initializes the covariance emulator in a warpper class by loading in the trained
        neural network. This is the class you should call during an analysis
        Class assumes the config yaml file used to train the network was saved to the same
        directory as the network itself
        @param net_dir {string} location of trained network and config file
        """

        self.config_dict = Dataset.load_config_file(net_dir+"config.yaml")

        # NOTE: Currently loads everything onto the cpu, which shouldn't matter
        # for using in an actual analysis
        self.net = Network_Emulator(self.config_dict).to(try_gpu())
        self.net.eval()
        self.net.load_state_dict(torch.load(net_dir+'network.params', map_location=torch.device("cpu")))


    def get_covariance_matrix(self, params, raw=False):
        """
        Uses the emulator to return a covariance matrix
        params -> secondary network -> decoder -> post-processing
        @param params {np array} the list of cosmology parameters to emulator a covariance matrix from
        @return C {np array} the emulated covariance matrix of size (N, N) where N was specified during initialization
        """
        params = torch.from_numpy(params).to(torch.float32)
        assert len(params) == self.config_dict.input_dim

        matrix = self.net(params.view(1,self.config_dict.input_dim)).view(1, self.config_dict.output_dim, self.config_dict.output_dim)

        if raw == False:
            matrix = Dataset.symmetric_exp(matrix, self.config_dict.norm_pos, self.config_dict.norm_neg).view(self.config_dict.output_dim,self.config_dict.output_dim)
            matrix = matrix.detach().numpy().astype(np.float64)
            matrix = np.matmul(matrix, matrix.T)
            return matrix
        else:
            return matrix

class Network_Emulator(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self.architecture = config_dict.architecture

        self.train_loss = []
        self.valid_loss = []
        self.num_epoch = []
        self.best_loss = 1e10

        self.save_state = None

        # input and putout dimensions
        self.input_dim = config_dict.input_dim
        self.output_dim = config_dict.output_dim
        self.N = torch.Tensor([config_dict.output_dim+1, config_dict.output_dim/2]).int()
        self.N_flat = self.N[0] * self.N[1]

        # Transformer variables
        self.patch_size = torch.Tensor(config_dict.patch_size)
        self.embedding = config_dict.embedding
        self.n_patches = (self.N / self.patch_size).int().tolist()
        self.patch_size = self.patch_size.int().tolist()
        sequence_len = int(self.patch_size[0]*self.patch_size[1])
        num_sequences = self.n_patches[0] * self.n_patches[1]

        # -----------------------------------------
        # MLP structure
        if self.architecture == "MLP":
            self.h1 = nn.Linear(config_dict.input_dim, config_dict.mlp_dims[0])
            self.mlp_blocks = nn.Sequential()
            for i in range(config_dict.num_mlp_blocks):
                self.mlp_blocks.add_module("ResNet"+str(i+1),
                        Blocks.Block_Full_ResNet(config_dict.mlp_dims[i],
                                                 config_dict.mlp_dims[i+1]))
            self.out = nn.Linear(config_dict.mlp_dims[-1], self.N_flat)

        # -----------------------------------------
        # MLP + Transformer structure
        elif self.architecture == "MLP-T":
            self.h1 = nn.Linear(config_dict.input_dim, config_dict.mlp_dims[0])
            self.mlp_blocks = nn.Sequential()
            for i in range(config_dict.num_mlp_blocks):
                self.mlp_blocks.add_module("ResNet"+str(i+1),
                        Blocks.Block_Full_ResNet(config_dict.mlp_dims[i],
                                                 config_dict.mlp_dims[i+1]))
            self.out = nn.Linear(config_dict.mlp_dims[-1], self.N_flat)

            self.linear_map = nn.Linear(sequence_len, sequence_len)
            self.transform_blocks = nn.Sequential()
            for i in range(config_dict.num_transformer_blocks):
                self.transform_blocks.add_module("transform"+str(i+1), 
                    nn.TransformerEncoderLayer(sequence_len, config_dict.num_heads, 4*sequence_len,
                                               config_dict.dropout_prob, "gelu", batch_first=True))
                   #Blocks.Block_Transformer_Encoder(sequence_len, num_heads, dropout_prob))
            #self.out2 = nn.Linear(2*sequence_len, sequence_len)

            if self.embedding == True:
                pos_embed = self.get_positional_embedding(num_sequences, sequence_len).to(try_gpu())
                pos_embed.requires_grad = False
                self.register_buffer("pos_embed", pos_embed)

            self.pos_embed = self.get_positional_embedding(num_sequences, sequence_len).to(try_gpu())
            self.pos_embed.requires_grad = False
        else:
            print("ERROR! Invalid architecture specified")

        bounds = torch.tensor(config_dict.parameter_bounds).to(try_gpu())
        self.register_buffer("bounds", bounds)

    def load_pretrained(self, path, freeze=True):
        """
        loads the pre-trained layers into the current model
        """
        pre_trained_dict = torch.load(path, map_location=try_gpu())

        for name, param in pre_trained_dict.items():
            if name not in self.state_dict():
                continue
            self.state_dict()[name].copy_(param)
            if freeze==True: self.state_dict()[name].requires_grad = False

        #print("Pre-trained layers loaded in succesfully")

    def normalize(self, params):
        """
        Normalizes the input parameters to a range of (0, 1)
        """
        params_norm = (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        return params_norm

    def normal(m):
        """
        Function that lets you randomly set your initial network weights based on 
        a normal distribution
        """
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=0., std=0.1)
            nn.init.zeros_(m.bias)

    def xavier(m):
        """
        Function that lets you randomly set your initial network weights based on 
        the xavier method: https://pytorch.org/docs/stable/nn.init.html
        """
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def He(m):
        """
        Function that lets you randomly set your initial network weights 
        Using the method from He et al 2015
        """
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)

    def get_positional_embedding(self, sequence_length, d):
        """
        Defines a position-dependent function to be added to your input when using
        a transformer network
        @param sequence_length {int} the length of each independent sequence in the input
        @param d {int} the number of sequences in your input
        @return embeddings {2D Tensor} the positional embedding to be applied
        """

        embeddings = torch.ones(sequence_length, d)

        for i in range(sequence_length):
            for j in range(d):
                embeddings[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j- 1 ) / d)))
        return embeddings

    def patchify(self, X):
        """
        Takes an input tensor and splits it into 2D patches before flattening each patch to 1D again
        @param X {3D Tensor} Covariance Matrix with shape (batch size, 51, 25)
        @return patches {3D Tensor} flattened patches of the input
        with shape (batch size, number of patches, size of patch)
        """
        X = X.reshape(-1, self.N[0], self.N[1])
        patches = X.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        patches = patches.reshape(-1, self.n_patches[1]*self.n_patches[0], self.patch_size[0]*self.patch_size[1])
        #patches = patches.permute(0, 2, 1)
        #print(patches.shape)
        return patches

    def un_patchify(self, patches):
        """
        Combines images patches (stored as 1D tensors) into a full image
        @param patches {4D Tensor} Transformer block output of seperate covariance patches
        @return X {3D tensor} Full matrix batch created by combining adjacent patches together
        """

        #patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, self.n_patches[0], self.n_patches[1], self.patch_size[0], self.patch_size[1])
        X = patches.permute(0, 1, 3, 2, 4).contiguous()
        X = X.view(-1, self.N[0], self.N[1])
        return X

    def forward(self, X):
        """
        Steps through the network to go from input (cosmology parameters) to output (L matrix)
        @param X {2D Tensor} batch of cosmology parameters
        @return X {3D Tensor} batch of lower triangular matrices
        """

        X = self.normalize(X)

        if self.architecture == "MLP":
            X = F.leaky_relu(self.h1(X))
            for blk in self.mlp_blocks:
                X = F.leaky_relu(blk(X))
            X = torch.tanh(self.out(X))

            X = X.view(-1, self.N[0], self.N[1])
            X = Dataset.rearange_to_full(X, self.output_dim, True)
            return X

        elif self.architecture == "MLP-T":
            X = F.leaky_relu(self.h1(X))
            for blk in self.mlp_blocks:
                X = F.leaky_relu(blk(X))
            Y = torch.tanh(self.out(X))

            X = self.linear_map(self.patchify(Y))
            if self.embedding == True:
                pos_embed = self.pos_embed.repeat(Y.shape[0], 1, 1)
                X = X + pos_embed
            for blk in self.transform_blocks:
                X = blk(X)# + X
            X = torch.tanh(self.un_patchify(X)) + Y.view(-1, self.N[0], self.N[1])

            X = X.view(-1, self.N[0], self.N[1])
            X = Dataset.rearange_to_full(X, self.output_dim, True)
            return X

    def save(self, save_dir):
        """
        Saves the current network state and config parameters to file
        @param save_dir {string} the location to save the network at
        """
        training_data = torch.vstack([torch.Tensor(self.num_epoch), 
                                      torch.Tensor(self.train_loss), 
                                      torch.Tensor(self.valid_loss)])
        torch.save(training_data, save_dir+"train_data-"+self.architecture+".dat")
        with open(save_dir+'config.yaml', 'w') as outfile:
            yaml.dump(dict(self.config_dict), outfile, default_flow_style=False)

        # option to either save the current state, or some earlier checkpoint
        if self.save_state is None: 
            torch.save(self.state_dict(), save_dir+'network.params')
        else:
            torch.save(self.save_state, save_dir+'network.params')

    # ---------------------------------------------------------------------------
    # Training Loop
    # ---------------------------------------------------------------------------
    def Train(self, optimizer, train_data, valid_data,
              print_progress=True, save_dir="", iter=0):
        """
        Train the network!
        @param optimizer {torch.optim object} the optimization scheme to use (ex. Adam)
        @param train_data {MatrixDataset object} The data used to train the network
        @param valid_data {MatrixDataset object} The data used to test the network during training. Used to quantify if the network is overfitting
        @param print_progress {bool} whether or not to print the network performance to terminal
        @param save_dir {string} Location to dynamically save the network during training. If black, stores progress in save_state variable instead.
        @param iter {int} current training round (used for printing only)
        """
        # Keep track of the best validation loss for early stopping
        worse_epochs = 0
        beta = 0.

        #weights = ((torch.arange(0, 250) / 250.) + 1.).to(try_gpu())
        #weights = torch.flip(weights, (0,))

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config_dict.batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.config_dict.batch_size, shuffle=True, drop_last=True)

        for epoch in range(self.config_dict.num_epochs):
            # Run through the training set and update weights
            self.train()
            train_loss_sub, valid_loss_sub  = 0., 0.
            if self.architecture == "VAE": train_KLD_sub, valid_KLD_sub = 0., 0.
            
            for (i, batch) in enumerate(train_loader):
                # load data
                params = batch[0]
                if self.architecture == "MLP-PCA": matrix = batch[2]
                else: matrix = batch[1]

                # use network to get prediction
                if self.architecture == "VAE" or self.architecture == "AE":
                    prediction, mu, log_var = self.forward(matrix.view(self.config_dict.batch_size, self.output_dim, self.output_dim))
                    loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
                    train_KLD_sub += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
                elif self.architecture == "MLP-PCA":
                    prediction = self.forward(params.view(self.config_dictbatch_size, self.input_dim))
                    diff = abs(prediction - matrix)
                    loss = torch.sum(diff)
                else:
                    prediction = self.forward(params.view(self.config_dict.batch_size, self.input_dim))
                    loss = F.l1_loss(prediction, matrix, reduction="sum")

                assert torch.isnan(loss) == False 
                assert torch.isinf(loss) == False
                assert loss > 0

                # update model
                train_loss_sub += loss.item()
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e8)    
                optimizer.step()

            # run through the validation set
            self.eval()
            for (i, batch) in enumerate(valid_loader):
                params = batch[0]
                if self.architecture == "MLP-PCA": matrix = batch[2]
                else: matrix = batch[1]

                if self.architecture == "VAE" or self.architecture == "AE":
                    prediction, mu, log_var = self.forward(matrix.view(self.config_dict.batch_size, self.output_dim, self.output_dim))
                    loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
                    valid_KLD_sub  += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
                elif self.architecture == "MLP-PCA":
                    prediction = self.forward(params.view(self.config_dict.batch_size, self.input_dim))
                    diff = abs(prediction - matrix)
                    loss = torch.sum(diff)
                else:
                    prediction = self.forward(params.view(self.config_dict.batch_size, self.input_dim))
                    loss = F.l1_loss(prediction, matrix, reduction="sum")

                valid_loss_sub += loss.item()

            # Aggregate loss information
            self.num_epoch.append(epoch)
            self.train_loss.append(train_loss_sub / len(train_loader.dataset))
            self.valid_loss.append(valid_loss_sub / len(valid_loader.dataset))

            # save the network if the validation loss improved, else stop early if there hasn't been
            # improvement for several epochs
            if self.valid_loss[-1] < self.best_loss:
                self.best_loss = self.valid_loss[-1]
                if save_dir != "": self.save(save_dir)
                else: self.save_state = self.state_dict().copy()
                worse_epochs = 0
            else:
                worse_epochs+=1

            if print_progress == True: 
                print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, self.train_loss[-1], self.valid_loss[-1], worse_epochs))
                if beta != 0: print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(train_KLD_sub/len(train_loader.dataset), valid_KLD_sub/len(valid_loader.dataset)))

            if epoch > 15 and worse_epochs >= 15 and self.valid_loss[-1] > self.train_loss[-1]:
                if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
                break
        print("lr {:0.3e}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(
            self.config_dict.learning_rate[iter], self.config_dict.batch_size, self.best_loss, epoch - worse_epochs))
