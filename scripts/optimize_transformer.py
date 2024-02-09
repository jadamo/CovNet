import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

# This script trains the emulator for one value of each hyperparameter
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

import src as CovNet

# Total number of matrices in the training + validation + test set
N = 106000
#N = 10000

torch.set_default_dtype(torch.float32)

start_from_checkpoint = False
num_rounds = 3

# wether or not to train on just the gaussian covariance (this is a test)
train_gaussian_only = False
# wether to train the main and secondary nets
do_main = True

# flag to specify network structure
# 0 = fully-connected ResNet VAE
# 1 = CNN ResNet VAE
# 2 = Pure MLP
# 3 = AE
# 4 = MLP with Transformer Layers
architecture = "T"

#CovNet_dir = "/home/joeadamo/Research/CovNet/"
CovNet_dir = "/home/u12/jadamo/CovNet/"

training_dir = "/xdisk/timeifler/jadamo/Training-Set-HighZ-NGC/"
#training_dir = "./Data/Training-Set-HighZ-NGC/"

folder=architecture+"/"
if train_gaussian_only == True: folder = architecture+"-gaussian/"

save_dir = "./emulators/ngc_z3/"+folder
#save_dir = "/home/joeadamo/Research/CovNet/emulators/ngc_z3/"+folder

checkpoint_dir = "./emulators/ngc_z3/MLP/"

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
    print("Saving to", save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network structure flag =", architecture)

    if start_from_checkpoint == True:
        assert architecture == "MLP" or architecture == "MLP-T", "checkpointing only implimented for MLP-based networks!"

    batch_size = 700
    lr        = [1e-3, 1e-4, 1e-5]
    #lr        = 1.438e-4#0.0005#1.438e-3#8.859e-04

    #patch_sizes = torch.Tensor([[3, 5],
    #                            [3, 1],
    #                            [1, 5],
    #                            [17, 5],
    #                            [17, 25]]).int()
    patch_sizes = torch.Tensor([[1, 25]]).int()
    num_heads = torch.Tensor([1, 1, 1, 1, 1]).int()
    num_blocks = torch.Tensor([10, 12, 14, 16]).int()
    freeze_mlp = [True]

    loss_data = torch.zeros((len(patch_sizes), len(num_blocks), 2))

    # the maximum # of epochs doesn't matter so much due to the implimentation of early stopping
    num_epochs = 500

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # get the training / test datasets
    t1 = time.time()
    train_data = CovNet.MatrixDataset(training_dir, "training", 1., train_gaussian_only)
    valid_data = CovNet.MatrixDataset(training_dir, "validation", 1, train_gaussian_only)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))


    # Sanity test - load in the base MLP network and print its validation loss
    # net_MLP = CovNet.Network_Emulator("MLP", 0.25).to(CovNet.try_gpu()); net_MLP.eval()
    # net_MLP.load_pretrained(checkpoint_dir+"network.params", True)
    # loss_to_beat = 0.
    
    # for i in range(N_valid):
    #     params = valid_data[i][0].view(1, 6)
    #     matrix = valid_data[i][1].view(1, 50, 50).to(CovNet.try_gpu())
    #     predict = net_MLP(params).view(1, 50, 50)
    #     loss_to_beat += F.l1_loss(predict, matrix, reduction="sum").item()

    # ref_string = "Basic MLP Loss is "+f'{loss_to_beat / N_valid:0.3f}'
    # print(ref_string)

    best_loss = 1e10
    best_config = [0,0,0,0]

    ref_string=""
    save_str = ref_string+"\n"
    for patch in range(patch_sizes.shape[0]):
        for n in range(len(num_blocks)):

            for attempt in range(2):
                # initialize networks
                net = CovNet.Network_Emulator(architecture, 0.2, num_blocks[n].item(), 
                                            patch_sizes[patch], num_heads[patch], True).to(CovNet.try_gpu())
                net.apply(He)
                #net.load_pretrained(checkpoint_dir+"network.params", freeze_mlp[0])

                for round in range(num_rounds):

                    # Define the optimizer
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr[round])

                    # Train the network! Progress is saved to file within the function
                    net.Train(num_epochs, batch_size, \
                                    optimizer, train_data, valid_data, \
                                    False, "", lr=lr[round])

                loss_data[patch, n] = net.best_loss
                result_str="num_blocks = "+f'{num_blocks[n]:0d}' + \
                        ", patch size = " + str(patch_sizes[patch].tolist()) + \
                        ", best loss = " + f'{net.best_loss:0.3f}'
                print(result_str)
                save_str += result_str+"\n"

                if net.best_loss < best_loss:
                    best_loss = net.best_loss
                    best_config = [patch, n, attempt]
                    net.save(save_dir)

                # write results to file in case the run doesn't finish fully
                torch.save(loss_data, save_dir+"output_config.dat")
                with open(save_dir+"output_config.log", "w") as file:
                    file.write(save_str)

    print("Best loss was {:0.3f} for the following configureation".format(best_loss))
    print("patch = " + str(patch_sizes[best_config[0]]) + ", num_blocks = " + str(best_config[1]) + \
        ", freeze mlp weights = " + str(freeze_mlp[best_config[2]]) + ", attempt " + str(best_config[3]))
if __name__ == "__main__":
    main()
