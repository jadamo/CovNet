import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import src.Blocks as Blocks
import src.Dataset as Dataset
from src.Dataset import try_gpu

# ---------------------------------------------------------------------------
# Emulator API Class - this is the class to call in a likelihood analysis
# ---------------------------------------------------------------------------
class CovNet():

    def __init__(self, net_dir, N, architecture=0,
                 pos_norm=5.91572, neg_norm=4.62748):
        """
        Initializes the covariance emulator in a warpper class by loading in the trained
        neural networks based on the specified options
        @param net_dir {string} location of trained networks
        @param N {int} the matrix dimensionality
        @param architecture {int} flag specifying the specific structure the network is (0 = fully connected ResNet, 1 = CNN ResNet)
        @param pos_norm {float} the normalization value to be applied to positive elements of each matrix
        @param neg_norm {float} the normalization value to be applied to negative elements of each matrix
        """
        self.architecture = architecture
        self.N = N

        self.norm_pos = pos_norm
        self.norm_neg = neg_norm

        self.net = Network_Emulator(architecture).to(try_gpu())
        self.net.eval()
        self.net.load_state_dict(torch.load(net_dir+'network.params', map_location=torch.device("cpu")))
        
        if architecture == "VAE" or architecture == "AE":
            self.decoder = Blocks.Block_Decoder(architecture).to(try_gpu())
            self.decoder.eval()
            self.decoder.load_state_dict(self.net.Decoder.state_dict())

            self.net_latent = Blocks.Network_Latent().to(try_gpu())
            self.net_latent.load_state_dict(torch.load(net_dir+'network-latent.params', map_location=torch.device("cpu")))
            self.net_latent.load_state_dict(torch.load(net_dir+'network-latent.params', map_location=torch.device("cpu")))

    def get_covariance_matrix(self, params, raw=False):
        """
        Uses the emulator to return a covariance matrix
        params -> secondary network -> decoder -> post-processing
        @param params {np array} the list of cosmology parameters to emulator a covariance matrix from
        @return C {np array} the emulated covariance matrix of size (N, N) where N was specified during initialization
        """
        params = torch.from_numpy(params).to(torch.float32)
        if self.architecture == 0 or self.architecture == 3:
            z = self.net_latent(params).view(1,6)
            matrix = self.decoder(z).view(1,self.N,self.N)
        else:
            matrix = self.net(params.view(1,6)).view(1, self.N, self.N)

        if raw == False:
            matrix = Dataset.symmetric_exp(matrix, self.norm_pos, self.norm_neg).view(self.N,self.N)
            matrix = torch.matmul(matrix, torch.t(matrix))
            return matrix.detach().numpy().astype(np.float64)
        else:
            return matrix

class Network_Emulator(nn.Module):
    def __init__(self, architecture="MLP", dropout_prob=0.,
                 num_blocks=3, patch_size=torch.Tensor([3, 5]).int(), 
                 num_heads=1, embedding=True, quadrant=""):
        super().__init__()
        self.architecture = architecture
        self.embedding = embedding

        self.train_loss = []
        self.valid_loss = []
        self.num_epoch = []
        self.best_loss = 1e10

        self.save_state = None
        self.N = torch.Tensor([51, 25]).int()
        self.n_patches = (self.N / patch_size).int().tolist()
        self.patch_size = patch_size.tolist()
        sequence_len = int(self.patch_size[0]*self.patch_size[1])
        num_sequences = self.n_patches[0] * self.n_patches[1]
        self.quadrant = quadrant

        # VAE structure
        if architecture == "VAE" or architecture == "AE":
            self.Encoder = Blocks.Block_Encoder(architecture)
            self.Decoder = Blocks.Block_Decoder(architecture)
        # MLP structure
        elif architecture == "MLP":
            self.h1 = nn.Linear(6, 25)
            self.resnet1 = Blocks.Block_Full_ResNet(25, 50)
            self.resnet2 = Blocks.Block_Full_ResNet(50, 100)
            self.resnet3 = Blocks.Block_Full_ResNet(100, 500)
            self.resnet4 = Blocks.Block_Full_ResNet(500, 1000)
            self.out = nn.Linear(1000, 51*25)
        # MLP + Transformer structure
        elif architecture == "MLP-T":
            self.h1 = nn.Linear(6, 25)
            self.resnet1 = Blocks.Block_Full_ResNet(25, 50)
            self.resnet2 = Blocks.Block_Full_ResNet(50, 100)
            self.resnet3 = Blocks.Block_Full_ResNet(100, 500)
            self.resnet4 = Blocks.Block_Full_ResNet(500, 1000)
            self.out = nn.Linear(1000, 51*25)

            self.linear_map = nn.Linear(sequence_len, sequence_len)
            self.transform_blocks = nn.Sequential()
            for i in range(num_blocks):
                #self.transform_blocks.add_module("linear"+str(i+1),
                #nn.Sequential(nn.Linear(sequence_len, sequence_len),
                #              nn.LeakyReLU(),
                #              nn.Linear(sequence_len, sequence_len),
                #              nn.LeakyReLU(),
                #              nn.Linear(sequence_len, sequence_len),
                #              nn.LeakyReLU(),
                #              nn.Linear(sequence_len, sequence_len)))
                self.transform_blocks.add_module("transform"+str(i+1), 
                    nn.TransformerEncoderLayer(sequence_len, num_heads, 4*sequence_len,
                                               dropout_prob, "gelu", batch_first=True))
                   #Blocks.Block_Transformer_Encoder(sequence_len, num_heads, dropout_prob))
            #self.out2 = nn.Linear(2*sequence_len, sequence_len)

            if self.embedding == True:
                self.pos_embed = self.get_positional_embedding(num_sequences, sequence_len).to(try_gpu())
                self.pos_embed.requires_grad = False
        # MLP emulating quadrants
        elif architecture == "MLP-Quadrants":
            self.h1 = nn.Linear(6, 25)
            self.resnet1 = Blocks.Block_Full_ResNet(25, 75)
            self.resnet2 = Blocks.Block_Full_ResNet(75, 250)
            self.resnet3 = Blocks.Block_Full_ResNet(250, 500)
            self.resnet4 = Blocks.Block_Full_ResNet(500, 625)
            self.out = nn.Linear(625, 625)
        # MLP emulating PCs
        elif architecture == "MLP-PCA":
            self.h1 = nn.Linear(6, 25)
            self.resnet1 = Blocks.Block_Full_ResNet(25, 75)
            self.resnet2 = Blocks.Block_Full_ResNet(75, 200)
            self.resnet3 = Blocks.Block_Full_ResNet(200, 200)
            self.resnet4 = Blocks.Block_Full_ResNet(200, 200)
            self.out = nn.Linear(200, 200)
        else:
            print("ERROR! Invalid architecture specified")

        self.bounds = torch.tensor([[50, 100],
                                [0.01, 0.3],
                                [0.25, 1.65],
                                [1, 4],
                                [-4, 4],
                                [-4, 4]]).to(try_gpu())

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

        params_norm = (params - self.bounds[:,0]) / (self.bounds[:,1] - self.bounds[:,0])
        return params_norm

    def get_positional_embedding(self, sequence_length, d):

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
        X = X.reshape(-1, 51, 25)
        patches = X.unfold(1, self.patch_size[0], self.patch_size[0]).unfold(2, self.patch_size[1], self.patch_size[1])
        patches = patches.reshape(-1, self.n_patches[1]*self.n_patches[0], self.patch_size[0]*self.patch_size[1])
        
        #patches = patches.permute(0, 2, 1)
        #print(patches.shape)
        return patches

    def un_patchify(self, patches):
        """
        Combines images patches (stored as 1D tensors) into a full image
        """

        #patches = patches.permute(0, 2, 1)
        patches = patches.reshape(-1, self.n_patches[0], self.n_patches[1], self.patch_size[0], self.patch_size[1])
        X = patches.permute(0, 1, 3, 2, 4).contiguous()
        X = X.view(-1, self.N[0], self.N[1])
        return X

    def forward(self, X):
        
        if self.architecture == "VAE" and self.architecture == "AE":
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

        elif self.architecture == "MLP":
            X = self.normalize(X)
            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            X = torch.tanh(self.out(X))

            X = X.view(-1, 51, 25)
            X = Dataset.rearange_to_full(X, 50, True)
            return X

        elif self.architecture == "MLP-T":
            X = self.normalize(X)
            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            Y = torch.tanh(self.out(X))

            # X = F.leaky_relu(self.extra1(Y))
            # X = F.leaky_relu(self.extra2(X))
            # X = F.leaky_relu(self.extra3(X))
            # X = Y + torch.tanh(self.extra4(X))
            X = self.linear_map(self.patchify(Y))
            if self.embedding == True:
                pos_embed = self.pos_embed.repeat(Y.shape[0], 1, 1)
                X = X + pos_embed
            for blk in self.transform_blocks:
                X = blk(X)# + X
            X = torch.tanh(self.un_patchify(X)) + Y.view(-1, 51, 25)
            # for blk in self.MLP_block:
            #     X = blk(X)

            X = X.view(-1, 51, 25)
            X = Dataset.rearange_to_full(X, 50, True)
            return X

        elif self.architecture == "MLP-Quadrants":
            X = self.normalize(X)
            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            X = torch.tanh(self.out(X))

            X = X.view(-1, 25, 25)
            if self.quadrant == "00" or self.quadrant == "22":
                X = torch.tril(X)
            return X
        elif self.architecture == "MLP-PCA":
            X = self.normalize(X)
            X = F.leaky_relu(self.h1(X))
            X = self.resnet1(X)
            X = self.resnet2(X)
            X = self.resnet3(X)
            X = self.resnet4(X)
            X = F.leaky_relu(self.out(X))
            return X

    def save(self, save_dir):
        training_data = torch.vstack([torch.Tensor(self.num_epoch), 
                                      torch.Tensor(self.train_loss), 
                                      torch.Tensor(self.valid_loss)])
        torch.save(training_data, save_dir+"train_data"+self.quadrant+".dat")
        if self.save_state is None: 
            torch.save(self.state_dict(), save_dir+'network'+self.quadrant+'.params')
        else:
            torch.save(self.save_state, save_dir+'network'+self.quadrant+'.params')

    # ---------------------------------------------------------------------------
    # Training Loops
    # ---------------------------------------------------------------------------
    def Train(self, num_epochs, batch_size,
              optimizer, train_data, valid_data,
              print_progress=True, save_dir="", lr=0, beta=0):
        """
        Train the pure MLP network
        """
        # Keep track of the best validation loss for early stopping
        worse_epochs = 0

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(num_epochs):
            # Run through the training set and update weights
            self.train()
            train_loss_sub, valid_loss_sub  = 0., 0.
            if self.architecture == "VAE": train_KLD_sub, valid_KLD_sub = 0., 0.
            
            for (i, batch) in enumerate(train_loader):
                # load data
                params = batch[0]
                if self.architecture == "MLP-Quadrants": matrix = train_data.get_quadrant(batch[2], self.quadrant)
                elif self.architecture == "MLP-PCA": matrix = batch[2]
                else: matrix = batch[1]

                # use network to get prediction
                if self.architecture == "VAE" or self.architecture == "AE":
                    prediction, mu, log_var = self.forward(matrix.view(batch_size, 50, 50))
                    loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
                    train_KLD_sub += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
                else:
                    prediction = self.forward(params.view(batch_size, 6))
                    loss = F.l1_loss(prediction, matrix, reduction="sum")

                assert torch.isnan(loss) == False 
                assert torch.isinf(loss) == False

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
                if self.architecture == "MLP-Quadrants": matrix = valid_data.get_quadrant(batch[2], self.quadrant)
                elif self.architecture == "MLP-PCA": matrix = batch[2]
                else: matrix = batch[1]

                if self.architecture == "VAE" or self.architecture == "AE":
                    prediction, mu, log_var = self.forward(matrix.view(batch_size, 50, 50))
                    loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
                    valid_KLD_sub  += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
                else:
                    prediction = self.forward(params.view(batch_size, 6))
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
        print("lr {:0.3e}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(lr, batch_size, self.best_loss, epoch - worse_epochs))


def train_latent(net, num_epochs, optimizer, train_loader, valid_loader,
                    print_progress=True, save_dir=""):
    """
    Train the features network
    """
    best_loss = 1e10
    worse_epochs = 0
    net_save = Blocks.Network_Latent()

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction = net(params)
            loss = Dataset.features_loss(prediction, features)
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
            loss = Dataset.features_loss(prediction, features)
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
