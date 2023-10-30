import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math
from easydict import EasyDict

# This script trains the emulator for several different sizes of the training set
# in order to investigate how training set size affects preformance
# NOTE: This script assumes you want to train the MLP block first seperately

import src as CovNet

torch.set_default_dtype(torch.float32)

config_dir = "/home/u12/jadamo/CovNet/config-files/covnet_BOSS_hpc.yaml"
# directory to save the intermediate MLP network
mlp_save_dir = "/home/u12/jadamo/CovNet/emulators/ngc_z3/MLP/"
output_dir = "/home/u12/jadamo/CovNet/emulators/ngc_z3"

save_dir_1 = "/home/u12/jadamo/CovNet/emulators/ngc_z3/MLP-T-001/"
save_dir_2 = "/home/u12/jadamo/CovNet/emulators/ngc_z3/MLP-T-005/"
save_dir_3 = "/home/u12/jadamo/CovNet/emulators/ngc_z3/MLP-T-01/"
save_dir_4 = "/home/u12/jadamo/CovNet/emulators/ngc_z3/MLP-T/"
save_dirs = [save_dir_1, save_dir_2, save_dir_3, save_dir_4]

num_attempts = 2

def get_testing_loss(net, test_data):
    avg_loss = 0
    for i in range(len(test_data)):
        params = test_data[i][0].view(1, net.config_dict.input_dim)
        matrix = test_data[i][1].view(1, net.config_dict.output_dim, net.config_dict.output_dim).to(CovNet.try_gpu())
        prediction = net(params)
        avg_loss += F.l1_loss(prediction, matrix, reduction="sum")
    avg_loss /= len(test_data)
    return avg_loss

def main():

    config_dict = CovNet.load_config_file(config_dir)

    print("Training with just gaussian term:   " + str(config_dict.train_gaussian_only))
    print("Using GPU:", torch.cuda.is_available())
    print("network architecture =", config_dict.architecture)

    assert config_dict.architecture == "MLP-T", "This file only works with the full network configuration!"
    assert config_dict.train_mlp_first == True

    data_sizes = torch.Tensor([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    batch_sizes = torch.Tensor([250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 400, 500, 500, 600]).int()
    num_rounds = len(config_dict.learning_rate)
    save_sizes = [0, 1, 4, 13]; idx = 0 

    # get the training / test datasets
    t1 = time.time()
    valid_data = CovNet.MatrixDataset(config_dict.training_dir, "validation", 1, 
                                      config_dict.train_gaussian_only,
                                      config_dict.norm_pos, config_dict.norm_neg)
    test_data = CovNet.MatrixDataset(config_dict.training_dir, "testing", 1, 
                                      config_dict.train_gaussian_only,
                                      config_dict.norm_pos, config_dict.norm_neg)
    t2 = time.time()
    print("Done loading in validation + test data, took {:0.2f} s".format(t2 - t1))
    
    loss_data = torch.zeros((data_sizes.shape[0], num_attempts, 4))
    time_data = torch.zeros((data_sizes.shape[0], num_attempts, 2))
    save_str = ""
    for size in range(data_sizes.shape[0]):

        train_data = CovNet.MatrixDataset(config_dict.training_dir, "training", data_sizes[size], 
                                          config_dict.train_gaussian_only, 
                                          config_dict.norm_pos, config_dict.norm_neg)
        print(train_data.matrices.shape[0])

        config_dict_MLP = EasyDict(config_dict)
        config_dict_MLP.architecture = "MLP"
        config_dict_MLP.batch_size = batch_sizes[size]
        config_dict_MLP.save_dir = mlp_save_dir

        # ----------------------------------------------------------
        # Train the MLP Block
        # ----------------------------------------------------------
        best_loss = 100000
        for attempt in range(num_attempts):
            # initialize networks
            net = CovNet.Network_Emulator(config_dict_MLP).to(CovNet.try_gpu())
            net.apply(CovNet.Network_Emulator.He)

            t1 = time.time()
            for round in range(num_rounds):

                # Define the optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=config_dict_MLP.learning_rate[round])

                # Train the network! Progress is saved to file within the function
                net.Train(optimizer, train_data, valid_data, \
                          False, "", round)

            t2 = time.time()
            test_loss = get_testing_loss(net, test_data)
            loss_data[size, attempt, 0] = net.best_loss
            loss_data[size, attempt, 1] = test_loss
            time_data[size, attempt, 0] = t2 - t1

            result_str="data fraction = "+f'{data_sizes[size]:0.2f}' + \
                    ", total matrixes = "+f'{train_data.matrices.shape[0]:0.0f}' + \
                    ", best validation loss = " + f'{net.best_loss:0.3f}' + \
                    ", test-set loss = " + f'{test_loss:0.3f}'
            print(result_str)
            save_str += result_str+"\n"
            #save_data = torch.vstack([loss_data, time_data])
            #torch.save(save_data, save_dir+"output_data.dat")

            # save the best run out of the attempts
            if net.best_loss < best_loss:
                best_loss = net.best_loss
                net.save(config_dict_MLP.save_dir)

            # write results to file in case the run doesn't finish fully
            with open(output_dir+"output_data.log", "w") as file:
                file.write(save_str)

        mlp_str = "Best MLP loss was " + f"{best_loss:0.3f}"
        print(mlp_str)
        save_str += mlp_str+"\n"

        # ----------------------------------------------------------
        # Train the Transformer Block
        # ----------------------------------------------------------
        best_loss = 100000
        for attempt in range(num_attempts):
            # initialize networks

            config_dict_full = EasyDict(config_dict)
            config_dict_full.batch_size = batch_sizes[size]

            net = CovNet.Network_Emulator(config_dict_full).to(CovNet.try_gpu())
            net.apply(CovNet.Network_Emulator.He)
            net.load_pretrained(mlp_save_dir+"network.params", config_dict.freeze_mlp)

            t1 = time.time()
            for round in range(num_rounds):

                # Define the optimizer
                optimizer = torch.optim.Adam(net.parameters(), lr=config_dict.learning_rate[round])

                # Train the network! Progress is saved to file within the function
                net.Train(optimizer, train_data, valid_data, \
                          False, "", round)

            if net.best_loss < best_loss and size in save_sizes:
                best_loss = net.best_loss
                net.save(save_dirs[idx])

            test_loss = get_testing_loss(net, test_data)
            loss_data[size, attempt, 2] = net.best_loss
            loss_data[size, attempt, 3] = test_loss
            time_data[size, attempt, 1] = t2 - t1
            result_str="data fraction = "+f'{data_sizes[size]:0.2f}' + \
                ", total matrixes = "+f'{train_data.matrices.shape[0]:0.0f}' + \
                ", best validation loss = " + f'{net.best_loss:0.3f}' + \
                ", test-set loss = " + f'{test_loss:0.3f}'
            print(result_str)

            save_str += result_str+"\n"
            with open(output_dir+"output_data.log", "w") as file:
                file.write(save_str)

            save_data = torch.vstack([loss_data, time_data])
            torch.save(save_data, output_dir+"output_data.dat")

        if size in save_sizes: idx+=1

    print("Best loss was {:0.3f}".format(best_loss))

if __name__ == "__main__":
    main()
