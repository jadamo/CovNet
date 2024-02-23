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
    """Class defining the covariance matrix emulator for external use."""

    def __init__(self, net_dir:str):
        """Initializes the covariance emulator 
        
        This is a warpper class that loads in the trained
        neural network, and is what you should call during an analysis. Class 
        assumes the config yaml file used to train the network was saved to the same
        directory as the network itself
        
        Args:
            net_dir: location of trained network and config file
        """

        self.config_dict = Dataset.load_config_file(net_dir+"config.yaml")

        # NOTE: Currently loads everything onto the cpu, which shouldn't matter
        # for using in an actual analysis
        self.net = Network_Emulator(self.config_dict).to(try_gpu())
        self.net.eval()
        self.net.load_state_dict(torch.load(net_dir+'network.params', map_location=torch.device("cpu")))


    def get_covariance_matrix(self, params, raw=False):
        """Uses the emulator to return a covariance matrix

        Args:
            params: The list of cosmology parameters to generate a covariance matrix from.
            raw: If True, returns the matrix without reversing pre-processing steps.\
            Default False
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

    def get_avg_loss(self, data):
        """Helpfer function to get the average loss from a given dataset.
        
        Args:
            data: the dataset (as a MatrixDataset object) to generate loss values for
        """
        return Dataset.get_avg_loss(self.net, data)


class Network_Emulator(nn.Module):
    """
    Class defining the neural network used to emulate covariance matrices.
    Includes functions for saving, loading, training, and using the network.
    In an actual analysis, you should use the CovNet class instead, which wraps around
    This one for simplicity
    """

    def __init__(self, config_dict):
        """Inititalizes the neural network based on the input configuration file.
        
        Args:
            config_dict: easydict dictionary of parameters specifying how \
            to build the network
        """
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

        # initialize weights
        self.apply(self.init_weights)

    def init_weights(self, m):
        """Initializes weights using a specific scheme set in the input yaml file
        
        This function is meant to be called by the constructor only.
        Current options for initialization schemes are ["normal", "He", "xavier"]
        """
        if isinstance(m, nn.Linear):
            if self.config_dict.weight_initialization == "He":
                nn.init.kaiming_uniform_(m.weight)
            elif self.config_dict.weight_initialization == "normal":
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)
            elif self.config_dict.weight_initialization == "xavier":
                nn.init.xavier_normal_(m.weight)
            else: # if scheme is invalid, use normal initialization as a substitute
                nn.init.normal_(m.weight, mean=0., std=0.1)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, path:str, freeze=True):
        """loads the pre-trained layers from a file into the current model
        
        Args:
            path: The directory+filename of the trained network to load
            freeze: If True, freezes loaded-in weights to their current values. \
            Default True
        """
        pre_trained_dict = torch.load(path, map_location=try_gpu())

        for name, param in pre_trained_dict.items():
            if name not in self.state_dict():
                continue
            self.state_dict()[name].copy_(param)
            if freeze==True: self.state_dict()[name].requires_grad = False


    def normalize(self, params):
        """Normalizes the input parameters to a range of (0, 1)
        
        This function requires you properly specify parameter bounds 
        in the config file used to initialize the network.
        
        Args:
            params: batch of input parameters to normalize (2D tensor)
        """
        params_norm = (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        return params_norm


    def get_positional_embedding(self, sequence_length:int, d:int):
        """Returns a position-dependent function to be added to your input when using
        a transformer network
        
        Args:
            sequence_length: The length of each independent sequence in the input
            d: The number of sequences in your input
        """

        embeddings = torch.ones(sequence_length, d)

        for i in range(sequence_length):
            for j in range(d):
                embeddings[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j- 1 ) / d)))
        return embeddings

    def patchify(self, X):
        """Splits input into specially-adjacent patches that are then flattened
        
        Args:
            X: Covariance Matrix with shape [batch size, N+1, (N+1)/2]
        Returns:
            patches flattened patches of the input with shape \
            [batch size, number of patches, size of patch]
        """
        X = X.reshape(-1, self.N[0], self.N[1])
        patches = X.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        patches = patches.reshape(-1, self.n_patches[1]*self.n_patches[0], self.patch_size[0]*self.patch_size[1])

        return patches

    def un_patchify(self, patches):
        """Combines image patches (stored as 1D tensors) into a full matrix
        
        Args:
            patches: Transformer block output of seperate covariance patches
        Returns:
            X: Full matrix batch created by combining adjacent patches together
        """

        #patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, self.n_patches[0], self.n_patches[1], self.patch_size[0], self.patch_size[1])
        X = patches.permute(0, 1, 3, 2, 4).contiguous()
        X = X.view(-1, self.N[0], self.N[1])
        return X

    def forward(self, X):
        """Steps through the network 
        
        Specifically, this function go from input (cosmology parameters) 
        to output (L matrix) based on the architecture defined in self.config_dict

        Args:
            X: 2D Tensor batch of cosmology parameters with size [batch_size, num_params]
        Returns:
            X: 3D Tensor batch of lower triangular matrices with size [batch_size, N, N]
        """

        X = self.normalize(X)

        if self.architecture == "MLP":
            X = F.leaky_relu(self.h1(X))
            for blk in self.mlp_blocks:
                X = F.leaky_relu(blk(X))
            X = torch.tanh(self.out(X))

            X = X.view(-1, self.N[0], self.N[1])
            X = Dataset.rearange_to_full(X, True)
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
            X = Dataset.rearange_to_full(X, True)
            return X

    def save(self, save_dir:str):
        """Saves the current network state and config parameters to file
        
        Args:
            save_dir: the location to save the network to
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
        """Train the network via minibatch stochastic gradient descent
        
        Args:
            optimizer: (torch.optim object) the optimization scheme to use (ex. Adam)
            train_data: (MatrixDataset object) The data used to train the network
            valid_data: (MatrixDataset object) The data used to test the network during training. Used to quantify if the network is overfitting
            print_progress: If True, prints training and validation loss to terminal. Default True
            save_dir: Location to dynamically save the network during training. \
                If "", stores progress in save_state variable instead. Detaulf ""
            iter: current training round (used for printing only)
        """

        worse_epochs = 0
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.config_dict.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=self.config_dict.batch_size, shuffle=False)

        for epoch in range(self.config_dict.num_epochs):
            # Run through the training set and update weights
            self.train()
            for (i, batch) in enumerate(train_loader):
                # load data
                params = batch[0]
                matrix = batch[1]

                # use network to get prediction
                prediction = self.forward(params)
                loss = F.l1_loss(prediction, matrix, reduction="sum")
                assert torch.isnan(loss) == False 
                assert torch.isinf(loss) == False
                assert loss > 0

                # update model
                #train_loss_sub += loss.item()
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1e8)    
                optimizer.step()

            self.eval()
            train_loss_sub, valid_loss_sub  = 0., 0.
            # To get accurate training loss data given the current net state, run thru
            # the training set again
            for params, matrix in train_loader:
                prediction = self.forward(params)
                loss = F.l1_loss(prediction, matrix, reduction="sum")
                train_loss_sub += loss.item()

            for params, matrix in valid_loader:
                prediction = self.forward(params)
                loss = F.l1_loss(prediction, matrix, reduction="sum")
                valid_loss_sub += loss.item()

            # Aggregate loss information
            self.num_epoch.append(epoch)
            self.train_loss.append(train_loss_sub / len(train_data))
            self.valid_loss.append(valid_loss_sub / len(valid_data))

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

            # early stopping criteria
            if self.config_dict.early_stopping_epochs != -1 and \
               worse_epochs >= self.config_dict.early_stopping_epochs  and \
               self.valid_loss[-1] > self.train_loss[-1]:
                if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
                break

        # re-load the save state or previous best network
        if save_dir != "":
            self.load_state_dict(torch.load(save_dir+'network.params', map_location=Dataset.try_gpu()))
        else: self.load_state_dict(self.save_state)

        print("lr {:0.3e}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(
            self.config_dict.learning_rate[iter], self.config_dict.batch_size, self.best_loss, epoch - worse_epochs))
