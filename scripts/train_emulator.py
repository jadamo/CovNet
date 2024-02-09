import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math, os
from easydict import EasyDict

# This script trains the emulator for one value of each hyperparameter
# All you need to specify is the path to your config file
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

from CovNet import Dataset, Emulator

torch.set_default_dtype(torch.float32)

# wether to train the main and secondary nets
do_main = True; do_features = False

config_file = "/home/joeadamo/Research/CovNet/config-files/covnet_BOSS_mac.yaml"
#config_file = "/home/u12/jadamo/CovNet/config-files/covnet_BOSS_hpc.yaml"

def main():

    config_dict = Dataset.load_config_file(config_file)

    print("Loading from Checkpoint             " + str(config_dict.start_from_checkpoint))
    print("Training with just gaussian term:   " + str(config_dict.train_gaussian_only))
    print("Saving to", config_dict.save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network architecture =", config_dict.architecture)

    lr = config_dict.learning_rate

    #num_epochs_latent = 250

    # initialize network(s)
    net = Emulator.Network_Emulator(config_dict).to(Dataset.try_gpu())
    net.apply(Emulator.Network_Emulator.He)

    if config_dict.start_from_checkpoint == True: 
        net.load_pretrained(config_dict.checkpoint_dir+"network.params", config_dict.freeze_mlp)

    # get the training / test datasets
    t1 = time.time()
    if "norm_pos" in net.config_dict:
        train_data = Dataset.MatrixDataset(config_dict.training_dir, "training", 1., 
                                           net.config_dict.train_gaussian_only,
                                           net.config_dict.norm_pos, net.config_dict.norm_neg)
    else:
        train_data = Dataset.MatrixDataset(config_dict.training_dir, "training", 1., 
                                           net.config_dict.train_gaussian_only,
                                           0, 0)
        norm_pos, norm_neg = train_data.norm_pos.item(), train_data.norm_neg.item()
        net.config_dict.norm_pos=norm_pos
        net.config_dict.norm_neg=norm_neg
    valid_data = Dataset.MatrixDataset(config_dict.training_dir, "validation", 1., 
                                       net.config_dict.train_gaussian_only, 
                                       net.config_dict.norm_pos, net.config_dict.norm_neg)
    
    print(net.config_dict.norm_pos, net.config_dict.norm_neg)
    
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    N_train = train_data.matrices.shape[0]
    N_valid = valid_data.matrices.shape[0]
    print(N_train)

    # -----------------------------------------------------------
    # Train the network
    # -----------------------------------------------------------

    # initial training round with just mlp block is specified as such
    if config_dict.train_mlp_first == True and config_dict.architecture == "MLP-T":
        print("Training mlp block first...")
        config_dict_mlp = EasyDict(config_dict)
        config_dict_mlp.architecture = "MLP"
        net_MLP = Emulator.Network_Emulator(config_dict_mlp).to(Dataset.try_gpu())
        net_MLP.apply(Emulator.Network_Emulator.He)

        for round in range(len(lr)):
            t1 = time.time()
            optimizer = torch.optim.Adam(net_MLP.parameters(), lr=lr[round])
            net_MLP.Train(optimizer, train_data, valid_data, \
                        True, config_dict.save_dir, round)
            t2 = time.time()
            print("Round {:0.0f} Done training network!, took {:0.0f} minutes {:0.2f} seconds\n".format(round+1, math.floor((t2 - t1)/60), (t2 - t1)%60))

        print("loading MLP block into full network...")
        net.load_pretrained(config_dict.save_dir+"network.params", config_dict.freeze_mlp)

    for round in range(len(lr)):

        t1 = time.time()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr[round])
        net.Train(optimizer, train_data, valid_data, \
                    True, config_dict.save_dir, round)
        t2 = time.time()
        print("Round {:0.0f} Done training network!, took {:0.0f} minutes {:0.2f} seconds\n".format(round+1, math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()
