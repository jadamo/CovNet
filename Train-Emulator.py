import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math, os

# This script trains the emulator for one value of each hyperparameter
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

import src as CovNet

torch.set_default_dtype(torch.float32)

start_from_checkpoint = False

# wether or not to train on just the gaussian covariance (this is a test)
train_gaussian_only = False
# wether to train the main and secondary nets
do_main = True; do_features = False

# flag to specify network structure, can be
# VAE           = fully-connected VAE
# AE            = AE
# MLP           = Pure MLP
# MLP-T         = MLP with Transformer Layers
# MLP-Quadrants = MLP emulating quadrants seperately
# MLP-PCA       = MLP emulating PCs
architecture = "MLP"
num_pcs = 250

if architecture != "VAE" and architecture != "AE": do_features = False

#CovNet_dir = "/home/joeadamo/Research/CovNet/"
CovNet_dir = "/home/u12/jadamo/CovNet/"

#training_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
training_dir = "/xdisk/timeifler/jadamo/Training-Set-HighZ-NGC/"
#training_dir = "./Data/Training-Set-HighZ-NGC/"

folder = architecture
if train_gaussian_only == True: folder += "-gaussian"
folder+="/"

save_dir = "./emulators/ngc_z3/"+folder
#save_dir = "/home/joeadamo/Research/CovNet/emulators/ngc_z3/"+folder

checkpoint_dir = "./emulators/ngc_z3/MLP/"

# parameter to control the importance of the KL divergence loss term
# A large value might result in posterior collapse
BETA = 0.01 if architecture == "VAE" else 0

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
    print("Loading from Checkpoint             " + str(start_from_checkpoint))
    print("Training with just gaussian term:   " + str(train_gaussian_only))
    print("Training main net: secondary net:    [" + str(do_main) + ", " + str(do_features) + "]")
    print("Saving to", save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network architecture =", architecture)

    if start_from_checkpoint == True:
        assert architecture == "MLP" or architecture == "MLP-T", "checkpointing only implimented for MLP-based networks!"

    batch_size = 400
    #lr        = [1e-2, 1e-3]
    lr        = [1.438e-3, 1e-4, 2e-5]
    #lr        = [1e-3, 1e-4, 1e-5]
    #lr        = 1.438e-4#0.0005#1.438e-3#8.859e-04
    lr_latent = 0.0035

    patch_size = torch.Tensor([3, 5]).int()
    embedding = True
    num_blocks = 3
    num_heads = 1
    freeze_mlp = False

    # the maximum # of epochs doesn't matter so much due to the implimentation of early stopping
    num_epochs = 275
    num_epochs_latent = 250

    # initialize networks
    if architecture != "MLP-Quadrants":
        net = CovNet.Network_Emulator(architecture, 0.25, 
                                  num_blocks, patch_size, num_heads, embedding).to(CovNet.try_gpu())
        net_latent = CovNet.Blocks.Network_Latent(False)
        net.apply(He)
        net_latent.apply(xavier)
    else:
        net00 = CovNet.Network_Emulator(architecture, 0.25, quadrant="00").to(CovNet.try_gpu())
        net22 = CovNet.Network_Emulator(architecture, 0.25, quadrant="22").to(CovNet.try_gpu())
        net02 = CovNet.Network_Emulator(architecture, 0.25, quadrant="02").to(CovNet.try_gpu())

        net00.apply(He)
        net22.apply(He)
        net02.apply(He)

    if start_from_checkpoint == True: 
        net.load_pretrained(checkpoint_dir+"network.params", freeze_mlp)

    # get the training / test datasets
    t1 = time.time()
    train_data = CovNet.MatrixDataset(training_dir, "training", 1., train_gaussian_only)
    valid_data = CovNet.MatrixDataset(training_dir, "validation", 1., train_gaussian_only)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    N_train = train_data.matrices.shape[0]
    N_valid = valid_data.matrices.shape[0]

    if architecture == "MLP-PCA":
        if os.path.exists(training_dir+"pca.pkl"):
            os.remove(training_dir+"pca.pkl")
        min_values, max_values = train_data.do_PCA(num_pcs, training_dir)
        valid_data.do_PCA(num_pcs, training_dir)

    for round in range(len(lr)):

        # Train the network! Progress is saved to file within the function
        if do_main:
            t1 = time.time()
            if architecture == "MLP-Quadrants":
                optimizer00 = torch.optim.Adam(net00.parameters(), lr=lr[round])
                optimizer22 = torch.optim.Adam(net22.parameters(), lr=lr[round])
                optimizer02 = torch.optim.Adam(net02.parameters(), lr=lr[round])
                net00.Train(num_epochs, batch_size, \
                              optimizer00, train_data, valid_data, \
                              True, save_dir, lr=lr[round])
                net22.Train(num_epochs, batch_size, \
                              optimizer22, train_data, valid_data, \
                              True, save_dir, lr=lr[round])
                net02.Train(num_epochs, batch_size, \
                              optimizer02, train_data, valid_data, \
                              True, save_dir, lr=lr[round])
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr[round])
                net.Train(num_epochs, batch_size, \
                          optimizer, train_data, valid_data, \
                          True, save_dir, lr=lr[round], beta=BETA)
            t2 = time.time()
            print("Round {:0.0f} Done training network!, took {:0.0f} minutes {:0.2f} seconds\n".format(round+1, math.floor((t2 - t1)/60), (t2 - t1)%60))

        # next, train the secondary network with the features from the VAE as the output
        if do_features:

            optimizer_latent = torch.optim.Adam(net_latent.parameters(), lr=lr_latent)

            t1 = time.time()
            # In case the network went thru early stopping, reload the net that was saved to file
            net.load_state_dict(torch.load(save_dir+'network.params', map_location=CovNet.try_gpu()))
            # separate encoder and decoders
            encoder = CovNet.Block_Encoder(architecture).to(CovNet.try_gpu())
            decoder = CovNet.Block_Decoder(architecture, True).to(CovNet.try_gpu())
            encoder.load_state_dict(net.Encoder.state_dict())
            decoder.load_state_dict(net.Decoder.state_dict())

            # gather feature data by running thru the trained encoder
            train_z = torch.zeros(N_train, 6, device=CovNet.try_gpu())
            valid_z = torch.zeros(N_valid, 6, device=CovNet.try_gpu())
            encoder.eval()
            for i in range(int(N_train / batch_size)):
                matrix = train_data[i*batch_size:(i+1)*batch_size][1].view(-1, 50, 50)
                z, mu, log_var = encoder(matrix)
                if architecture == 3: mu = z.detach()
                train_z[i*batch_size:(i+1)*batch_size, :] = mu.view(-1, 6).detach()
            for i in range(int(N_valid / batch_size)):
                matrix = valid_data[i*batch_size:(i+1)*batch_size][1].view(-1, 50, 50)
                z, mu, log_var = encoder(matrix)
                if architecture == 3: mu = z.detach()
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
            return_stuff = CovNet.Emulator.train_latent(net_latent, num_epochs_latent, \
                                                optimizer_latent, train_loader, valid_loader, \
                                                True, save_dir)
            t2 = time.time()
            print("Done training latent network!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    if architecture == "MLP-Quadrants":

        net00.load_state_dict(torch.load(save_dir+'network00.params', map_location=CovNet.try_gpu()))
        net22.load_state_dict(torch.load(save_dir+'network22.params', map_location=CovNet.try_gpu()))
        net02.load_state_dict(torch.load(save_dir+'network02.params', map_location=CovNet.try_gpu()))
        combined_loss = 0.
        for i in range(N_valid):
            params = valid_data[i][0].view(1, 6)
            matrix = valid_data[i][1].view(1, 50, 50).to(CovNet.try_gpu())
            C00 = net00(params).view(1, 25, 25)
            C22 = net22(params).view(1, 25, 25)
            C02 = net02(params).view(1, 25, 25)

            predict = CovNet.combine_quadrants(C00, C22, C02).view(1, 50, 50)
            combined_loss += F.l1_loss(predict, matrix, reduction="sum").item()

        print("Combined quadrants validation loss is {:0.3f}".format(combined_loss / N_valid))
    elif architecture == "MLP-PCA":
        net.load_state_dict(torch.load(save_dir+"network.params", map_location=CovNet.try_gpu()))
        combined_loss = 0.
        reconstruct_loss = 0.
        for i in range(N_valid):
            params = valid_data[i][0].view(1, 6)
            matrix = valid_data[i][1].view(1, 50, 50).to(CovNet.try_gpu())
            components = net(params).view(num_pcs)
            components_true = valid_data.components[i]
            predict = CovNet.reverse_pca(components, valid_data.pca, min_values, max_values)
            predict = predict.view(1, 51, 25)
            predict = CovNet.rearange_to_full(predict, 50, True)

            reconstruct = CovNet.reverse_pca(components_true, valid_data.pca, min_values, max_values)
            reconstruct = reconstruct.view(1, 51, 25)
            reconstruct = CovNet.rearange_to_full(reconstruct, 50, True)

            combined_loss += F.l1_loss(predict, matrix, reduction="sum").item()
            reconstruct_loss += F.l1_loss(reconstruct, matrix, reduction="sum").item()

        print("Comparison PCA validation loss is {:0.3f}".format(combined_loss / N_valid))
        print("Loss from  PCA reconstruction is {:0.3f}".format(reconstruct_loss / N_valid))

if __name__ == "__main__":
    main()
