import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

# This script trains the emulator for one value of each hyperparameter
# if you want to find the optimal value for your hyperparameters, you should run
# optimize-hyperparameters.py

import src as CovNet

torch.set_default_dtype(torch.float32)

CovNet_dir = "/home/joeadamo/Research/CovNet/"
num_attempts = 2

def main():

    config_file = CovNet.load_config_file(CovNet_dir+"config-files/covnet_BOSS.yaml")

    print("Loading from Checkpoint             " + str(config_file.start_from_checkpoint))
    print("Training with just gaussian term:   " + str(config_file.train_gaussian_only))
    print("Saving to", config_file.save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network architecture =", config_file.architecture)

    assert config_file.architecture == "MLP-T", "This file only works with the full network configuration!"

    data_sizes = torch.Tensor([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    batch_sizes = torch.Tensor([250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 400, 500, 500, 600]).int()
    num_rounds = len(config_file.learning_rate)
    
    # get the training / test datasets
    t1 = time.time()
    valid_data = CovNet.MatrixDataset(config_file.training_dir, "validation", 1, 
                                      config_file.train_gaussian_only, config_file.norm_pos, config_file.norm_neg)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    

    best_loss = 1e10
    loss_data = torch.zeros((data_sizes.shape[0], num_attempts, 2))
    time_data = torch.zeros((data_sizes.shape[0], num_attempts, 2))
    save_str = ""
    for size in range(data_sizes.shape[0]):

        train_data = CovNet.MatrixDataset(config_file.training_dir, "training", data_sizes[size], 
                                          config_file.train_gaussian_only, config_file.norm_pos, config_file.norm_neg)
        print(train_data.matrices.shape[0])

        temp_best_loss = 100000
        for attempt in range(num_attempts):
            # initialize networks
            net = CovNet.Network_Emulator("MLP").to(CovNet.try_gpu())
            net.apply(CovNet.Network_Emulator.He)

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
