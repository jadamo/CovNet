import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

# This script trains the emulator for one value of each hyperparameter
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
import CovNet

# Total number of matrices in the training + validation + test set
N = 106000
#N = 20000

torch.set_default_dtype(torch.float32)

# whether or not nuiscane parameters are varied in the training set
train_nuisance = False
# wether or not to train with the Cholesky decomposition
train_cholesky = True
# wether or not to train on just the gaussian covariance (this is a test)
train_gaussian_only = True
# wether to train the VAE and features nets
do_VAE = True; do_features = True

# flag to specify network structure
# 0 = fully-connected ResNet
# 1 = CNN ResNet
# 2 = Pure MLP (no VAE, just a simple fully connected network)
structure_flag = 2

if structure_flag == 2: do_features = False

training_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
#training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-HighZ-NGC/"

if structure_flag == 0: folder = "full"
elif structure_flag == 1: folder = "cnn"
elif structure_flag == 2: folder = "simple"
elif structure_flag == 3: folder = "AE"
if train_gaussian_only == True: folder += "-gaussian"
folder+="/"

save_dir = "/home/u12/jadamo/CovNet/emulators/ngc_z3/"+folder
#save_dir = "/home/joeadamo/Research/CovNet/emulators/ngc_z3/"+folder

# parameter to control the importance of the KL divergence loss term
# A large value might result in posterior collapse
BETA = 0.01 if structure_flag != 3 else 0

# Standard normal distribution
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.zeros_(m.bias)

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def He(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)

def main():

    #print("Training set varies nuisance parameters " + str(train_nuisance))
    print("Training with cholesky decomposition:   " + str(train_cholesky))
    print("Training with just gaussian term:       " + str(train_gaussian_only))
    print("Training VAE net: features net:        [" + str(do_VAE) + ", " + str(do_features) + "]")
    print("Saving to", save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network structure flag =", structure_flag)

    batch_size = 200
    lr_VAE    = 1.438e-3
    lr_latent = 0.0035

    # the maximum # of epochs doesn't matter so much due to the implimentation of early stopping
    num_epochs_VAE = 150
    num_epochs_latent = 250

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize networks
    net = CovNet.Network_Emulator(structure_flag, train_cholesky).to(CovNet.try_gpu())
    net_latent = CovNet.Network_Latent(train_nuisance)

    net.apply(He)
    net_latent.apply(xavier)

    # Define the optimizer
    optimizer_VAE = torch.optim.Adam(net.parameters(), lr=lr_VAE)
    optimizer_latent = torch.optim.Adam(net_latent.parameters(), lr=lr_latent)

    # get the training / test datasets
    t1 = time.time()
    train_data = CovNet.MatrixDataset(training_dir, N_train, 0, train_nuisance, \
                                      train_cholesky, train_gaussian_only)
    valid_data = CovNet.MatrixDataset(training_dir, N_valid, N_train, train_nuisance, \
                                      train_cholesky, train_gaussian_only)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    # Train the network! Progress is saved to file within the function
    if do_VAE:
        t1 = time.time()
        if structure_flag != 2: 
            return_stuff = CovNet.train_VAE(net, num_epochs_VAE, batch_size, BETA, structure_flag, \
                                            optimizer_VAE, train_loader, valid_loader, \
                                            True, save_dir, lr=lr_VAE)
        else: 
            return_stuff = CovNet.train_MLP(net, num_epochs_VAE, batch_size, structure_flag, \
                                            optimizer_VAE, train_loader, valid_loader, \
                                            True, save_dir, lr=lr_VAE)
        t2 = time.time()
        print("Done training VAE!, took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))


    # next, train the secondary network with the features from the VAE as the output
    if do_features:

        t1 = time.time()
        # In case the network went thru early stopping, reload the net that was saved to file
        net.load_state_dict(torch.load(save_dir+'network-VAE.params'))
        # separate encoder and decoders
        encoder = CovNet.Block_Encoder(structure_flag).to(CovNet.try_gpu())
        decoder = CovNet.Block_Decoder(structure_flag, train_cholesky).to(CovNet.try_gpu())
        encoder.load_state_dict(net.Encoder.state_dict())
        decoder.load_state_dict(net.Decoder.state_dict())

        # gather feature data by running thru the trained encoder
        train_z = torch.zeros(N_train, 6, device=CovNet.try_gpu())
        valid_z = torch.zeros(N_valid, 6, device=CovNet.try_gpu())
        encoder.eval()
        for i in range(int(N_train / batch_size)):
            matrix = train_data[i*batch_size:(i+1)*batch_size][1].view(-1, 50, 50)
            z, mu, log_var = encoder(matrix)
            train_z[i*batch_size:(i+1)*batch_size, :] = mu.view(-1, 6).detach()
        for i in range(int(N_valid / batch_size)):
            matrix = valid_data[i*batch_size:(i+1)*batch_size][1].view(-1, 50, 50)
            z, mu, log_var = encoder(matrix)
            valid_z[i*batch_size:(i+1)*batch_size, :] = mu.view(-1, 6).detach()

        # add feature data to the training set and reinitialize the data loaders
        train_data.add_latent_space(train_z)
        valid_data.add_latent_space(valid_z)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
        t2 = time.time()
        print("Done preparing latent network training set!, took {:0.2f} seconds".format(t2 - t1))

        # train the secondary network!
        t1 = time.time()
        return_stuff = CovNet.train_latent(net_latent, num_epochs_latent, \
                                            optimizer_latent, train_loader, valid_loader, \
                                            True, save_dir)
        t2 = time.time()
        print("Done training latent network!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()