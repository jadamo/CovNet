import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

# This script trains the emulator for one value of each hyperparameter
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

import src as CovNet

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
architecture = "MLP-T"

#CovNet_dir = "/home/joeadamo/Research/CovNet/"
CovNet_dir = "/home/u12/jadamo/CovNet/"

training_dir = "/xdisk/timeifler/jadamo/Training-Set-HighZ-NGC/"
#training_dir = "./Data/Training-Set-HighZ-NGC/"

folder=architecture+"/"
if train_gaussian_only == True: folder = architecture+"-gaussian/"

save_dir = "./emulators/ngc_z3/"+folder
#save_dir = "/home/joeadamo/Research/CovNet/emulators/ngc_z3/"+folder

checkpoint_dir = "./emulators/ngc_z3/MLP/"

save_dir_2 = "./emulators/ngc_z3/MLP-small/"
save_dir_3 = "./emulators/ngc_z3/MLP-T-small/"
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

    lr        = [1.438e-3, 1e-4, 2e-5]
    #lr        = 1.438e-4#0.0005#1.438e-3#8.859e-04

    data_sizes = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    batch_size = torch.Tensor([250, 250, 250, 250, 250, 250, 400, 500, 500, 600]).int()

    patch_size = torch.Tensor([17, 5]).int()
    num_blocks = 5
    num_heads = 5

    # the maximum # of epochs doesn't matter so much due to the implimentation of early stopping
    num_epochs = 300
    num_attempts = 2

    # get the training / test datasets
    t1 = time.time()
    valid_data = CovNet.MatrixDataset(training_dir, "validation", 1, train_gaussian_only)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    

    best_loss = 1e10
    loss_data = torch.zeros((data_sizes.shape[0], num_attempts, 2))
    time_data = torch.zeros((data_sizes.shape[0], num_attempts, 2))
    save_str = ""
    for size in range(data_sizes.shape[0]):

        train_data = CovNet.MatrixDataset(training_dir, "training", data_sizes[size], train_gaussian_only)
        print(train_data.matrices.shape[0])

        temp_best_loss = 100000
        for attempt in range(num_attempts):
            # initialize networks
            net = CovNet.Network_Emulator("MLP").to(CovNet.try_gpu())
            net.apply(He)

            t1 = time.time()
            for round in range(num_rounds):

                # Define the optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=lr[round])

                # Train the network! Progress is saved to file within the function
                net.Train(num_epochs, int(batch_size[size]), \
                                optimizer, train_data, valid_data, \
                                False, "", lr=lr[round])

            t2 = time.time()
            loss_data[size, attempt, 0] = net.best_loss
            time_data[size, attempt, 0] = t2 - t1
            if architecture == "MLP":
                result_str="data fraction = "+f'{data_sizes[size]:0.2f}' + \
                        ", total matrixes = "+f'{train_data.matrices.shape[0]:0.0f}' + \
                        ", best loss = " + f'{net.best_loss:0.3f}'
                if architecture == "MLP": print(result_str)
                save_str += result_str+"\n"
                save_data = torch.vstack([loss_data, time_data])
                torch.save(save_data, save_dir+"output_data.dat")

            if architecture == "MLP-T" and net.best_loss < temp_best_loss:
                temp_best_loss = net.best_loss
                net.save(checkpoint_dir)
                if size == 0: net.save(save_dir_2)
            elif architecture == "MLP" and net.best_loss < best_loss:
                best_loss = net.best_loss
                net.save(save_dir)

            # write results to file in case the run doesn't finish fully
            with open(save_dir+"output_data.log", "w") as file:
                file.write(save_str)

        # if we're testing the transofmer network, reload based off of the above loop result
        if architecture == "MLP-T":
            mlp_str = "Best MLP loss was " + f"{temp_best_loss:0.3f}"
            print(mlp_str)
            save_str += mlp_str+"\n"
            for attempt in range(num_attempts):
                # initialize networks
                net = CovNet.Network_Emulator(architecture, 0.2, num_blocks,
                                              patch_size, num_heads).to(CovNet.try_gpu())
                net.apply(He)
                net.load_pretrained(checkpoint_dir+"network.params", True)

                t1 = time.time()
                for round in range(num_rounds):

                    # Define the optimizer
                    optimizer = torch.optim.Adam(net.parameters(), lr=lr[round])

                    # Train the network! Progress is saved to file within the function
                    net.Train(num_epochs, int(batch_size[size]), \
                                    optimizer, train_data, valid_data, \
                                    False, "", lr=lr[round])

                if net.best_loss < best_loss:
                    best_loss = net.best_loss
                    net.save(save_dir)
                    if size == 0: net.save(save_dir_3)

                loss_data[size, attempt, 1] = net.best_loss
                time_data[size, attempt, 1] = t2 - t1
                result_str="data fraction = "+f'{data_sizes[size]:0.2f}' + \
                    ", total matrixes = "+f'{train_data.matrices.shape[0]:0.0f}' + \
                    ", best loss = " + f'{net.best_loss:0.3f}'
                print(result_str)
                save_str += result_str+"\n"
                with open(save_dir+"output_data.log", "w") as file:
                    file.write(save_str)

                save_data = torch.vstack([loss_data, time_data])
                torch.save(save_data, save_dir+"output_data.dat")

    print("Best loss was {:0.3f}".format(best_loss))

if __name__ == "__main__":
    main()
