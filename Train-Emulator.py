import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math, os
from easydict import EasyDict

# This script trains the emulator for one value of each hyperparameter
# All you need to specify is the path to your config file
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

import src as CovNet

torch.set_default_dtype(torch.float32)

# wether to train the main and secondary nets
do_main = True; do_features = False

CovNet_dir = "/home/joeadamo/Research/CovNet/"

def main():

    config_dict = CovNet.load_config_file(CovNet_dir+"config-files/covnet_BOSS.yaml")

    print("Loading from Checkpoint             " + str(config_dict.start_from_checkpoint))
    print("Training with just gaussian term:   " + str(config_dict.train_gaussian_only))
    print("Saving to", config_dict.save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network architecture =", config_dict.architecture)

    lr = config_dict.learning_rate

    #num_epochs_latent = 250

    # initialize network(s)
    net = CovNet.Network_Emulator(config_dict).to(CovNet.try_gpu())
    net.apply(CovNet.Network_Emulator.He)
    #net_latent = CovNet.Blocks.Network_Latent(False)
    #net_latent.apply(xavier)

    if config_dict.start_from_checkpoint == True: 
        net.load_pretrained(config_dict.checkpoint_dir+"network.params", config_dict.freeze_mlp)

    # get the training / test datasets
    t1 = time.time()
    if "norm_pos" in net.config_dict:
        train_data = CovNet.MatrixDataset(config_dict.training_dir, "training-small", 0.1, 
                                        net.config_dict.train_gaussian_only,
                                        net.config_dict.norm_pos, net.config_dict.norm_neg)
    else:
        train_data = CovNet.MatrixDataset(config_dict.training_dir, "training-small", 0.1, 
                                        net.config_dict.train_gaussian_only,
                                        0, 0)
        norm_pos, norm_neg = train_data.norm_pos.item(), train_data.norm_neg.item()
        net.config_dict.norm_pos=norm_pos
        net.config_dict.norm_neg=norm_neg
    valid_data = CovNet.MatrixDataset(config_dict.training_dir, "validation", 1., 
                                    net.config_dict.train_gaussian_only, 
                                    net.config_dict.norm_pos, net.config_dict.norm_neg)
    
    print(net.config_dict.norm_pos, net.config_dict.norm_neg)
    
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    N_train = train_data.matrices.shape[0]
    N_valid = valid_data.matrices.shape[0]
    print(N_train)

    # old code to setup pca emulator
    # if net.config_dict.architecture == "MLP-PCA":
    #     if os.path.exists(config_dict.training_dir+"pca.pkl"):
    #         os.remove(config_dict.training_dir+"pca.pkl")
    #     min_values, max_values = train_data.do_PCA(250, config_dict.training_dir)
    #     valid_data.do_PCA(250, config_dict.training_dir)

    # -----------------------------------------------------------
    # Train the network
    # -----------------------------------------------------------

    # initial training round with just mlp block is specified as such
    if config_dict.train_mlp_first == True and config_dict.architecture == "MLP-T":
        print("Training mlp block first...")
        config_dict_mlp = EasyDict(config_dict)
        config_dict_mlp.architecture = "MLP"
        net_MLP = CovNet.Network_Emulator(config_dict_mlp).to(CovNet.try_gpu())
        net_MLP.apply(CovNet.Network_Emulator.He)

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

    # Old code to test PCA emulator
    # if net.config_dict.architecture == "MLP-PCA":
    #     net.load_state_dict(torch.load(config_dict.save_dir+"network.params", map_location=CovNet.try_gpu()))
    #     combined_loss = 0.
    #     reconstruct_loss = 0.
    #     pc_error = 0.
    #     for i in range(N_valid):
    #         params = valid_data[i][0].view(1, 6)
    #         matrix = valid_data[i][1].view(1, 50, 50).to(CovNet.try_gpu())
    #         components = net(params).view(250)
    #         components_true = valid_data.components[i]
    #         predict = CovNet.reverse_pca(components, valid_data.pca, min_values, max_values)
    #         predict = predict.view(1, 51, 25)
    #         predict = CovNet.rearange_to_full(predict, 50, True)

    #         reconstruct = CovNet.reverse_pca(components_true, valid_data.pca, min_values, max_values)
    #         reconstruct = reconstruct.view(1, 51, 25)
    #         reconstruct = CovNet.rearange_to_full(reconstruct, 50, True)

    #         combined_loss += F.l1_loss(predict, matrix, reduction="sum").item()
    #         reconstruct_loss += F.l1_loss(reconstruct, matrix, reduction="sum").item()
    #         pc_error += torch.mean((components - components_true) / components_true)

    #     pc_error = 100 * pc_error / N_valid
    #     print("Comparison PCA validation loss is {:0.3f}".format(combined_loss / N_valid))
    #     print("Loss from  PCA reconstruction is {:0.3f}".format(reconstruct_loss / N_valid))
    #     print("Average error per PC = {:0.3f}".format(pc_error))

if __name__ == "__main__":
    main()
