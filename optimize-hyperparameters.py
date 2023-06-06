import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
import CovNet

# Total number of matrices in the training + validation + test set
N = 106000
#N = 10000

torch.set_default_dtype(torch.float32)

vary_learning_rate = True
vary_batch_size = False

# flag to specify network structure
# 0 = VAE fully-connected ResNet
# 1 = VAE CNN ResNet
# 2 = Pure MLP (no VAE, just a simple fully connected network)
# 3 = AE fully-connected ResNet
structure_flag = 3

training_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
#training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-HighZ-NGC/"

if structure_flag == 0: folder = "VAE"
elif structure_flag == 1: folder = "VAE-cnn"
elif structure_flag == 2: folder = "MLP"
elif structure_flag == 3: folder = "AE"
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

    print("Optimizing learning rate:", vary_learning_rate)
    print("Optimizing batch size:", vary_batch_size)
    print("Saving to", save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network structure flag =", structure_flag)

    if vary_learning_rate == True: lr_VAE = torch.logspace(-4, -2, 20).to(CovNet.try_gpu())
    else: lr_VAE = 1.438e-3#8.859e-4

    if vary_batch_size == True: batch_size = torch.Tensor([25, 50, 100, 200, 265, 424, 530]).to(torch.int)
    else: batch_size = 200
    lr_latent = 0.0035

    # the maximum # of epochs doesn't matter so much due to the implimentation of early stopping
    num_epochs_VAE = 150
    num_epochs_latent = 250

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # get the training / test datasets
    t1 = time.time()
    train_data = CovNet.MatrixDataset(training_dir, N_train, 0, False, \
                                      True, False)
    valid_data = CovNet.MatrixDataset(training_dir, N_valid, N_train, False, \
                                      True, False)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    iterate = len(lr_VAE) if vary_learning_rate == True else len(batch_size)
    lowest_loss = torch.zeros(iterate)
    lowest_loss_2 = torch.zeros(iterate)
    lowest_loss_3 = torch.zeros(iterate)

    try:
        lr_loss_dat = torch.load(save_str="optimized-lr.dat",map_location=CovNet.try_gpu())
        temp = lr_loss_dat[0,:][(lr_loss_dat[0,:]!= 0)]
        best_loss = torch.min(temp).item()
    except:
        best_loss = 1e5

    t1 = time.time()
    for i in range(iterate):

        # re-initialize networks
        net = CovNet.Network_Emulator(structure_flag, True).to(CovNet.try_gpu())
        net_latent = CovNet.Network_Latent(False)

        net.apply(He)
        net_latent.apply(xavier)

        lr = lr_VAE[i] if vary_learning_rate == True else lr_VAE
        bsize = batch_size[i].item() if vary_batch_size == True else batch_size

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=bsize, shuffle=True)

        # Define the optimizer
        optimizer_VAE = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer_latent = torch.optim.Adam(net_latent.parameters(), lr=lr_latent)
        
        # Train the network!
        if structure_flag != 2: 
            net, train_loss, valid_loss = \
                CovNet.train_VAE(net, num_epochs_VAE, bsize, BETA, structure_flag, \
                                 optimizer_VAE, train_loader, valid_loader, \
                                 False, lr=lr)
            lowest_loss_2[i] = torch.min(valid_loss[(valid_loss != 0)])
        else:
            net, train_loss, valid_loss = \
                CovNet.train_MLP(net, num_epochs_VAE, bsize, structure_flag, \
                                 optimizer_VAE, train_loader, valid_loader, \
                                 False, lr=lr)
            lowest_loss[i] = torch.min(valid_loss[(valid_loss != 0)])

        # next, train the secondary network with the features from the VAE as the output
        if structure_flag != 2:

            # separate encoder and decoders
            encoder = CovNet.Block_Encoder(structure_flag).to(CovNet.try_gpu())
            decoder = CovNet.Block_Decoder(structure_flag, True).to(CovNet.try_gpu())
            encoder.load_state_dict(net.Encoder.state_dict())
            decoder.load_state_dict(net.Decoder.state_dict())

            # gather feature data by running thru the trained encoder
            train_z = torch.zeros(N_train, 6, device=CovNet.try_gpu())
            valid_z = torch.zeros(N_valid, 6, device=CovNet.try_gpu())
            encoder.eval()
            for j in range(int(N_train / bsize)):
                matrix = train_data[j*bsize:(j+1)*bsize][1].view(-1, 50, 50)
                z, mu, log_var = encoder(matrix)
                train_z[j*bsize:(j+1)*bsize, :] = mu.view(-1, 6).detach()
            for j in range(int(N_valid / bsize)):
                matrix = valid_data[j*bsize:(j+1)*bsize][1].view(-1, 50, 50)
                z, mu, log_var = encoder(matrix)
                valid_z[j*bsize:(j+1)*bsize, :] = mu.view(-1, 6).detach()

            # add feature data to the training set and reinitialize the data loaders
            train_data.add_latent_space(train_z)
            valid_data.add_latent_space(valid_z)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=bsize, shuffle=True)

            # train the secondary network!
            net_latent, train_loss_2, valid_loss_2 = \
                CovNet.train_latent(net_latent, num_epochs_latent, \
                                    optimizer_latent, train_loader, valid_loader, \
                                    False)     
            lowest_loss_3[i] = torch.min(valid_loss_2[(valid_loss_2 != 0)])   

            # calculate the L1 loss from using the full emulator
            # this loss is the one we care about
            loss_full = 0
            for (j, batch) in enumerate(valid_loader):
                params = batch[0].to(torch.device("cpu")); matrix = batch[1]
                z = net_latent(params.view(bsize, 6)).to(CovNet.try_gpu())
                prediction = decoder(z).view(bsize, 50, 50)
                loss_full += F.l1_loss(prediction, matrix, reduction="sum").item()

            # Aggregate loss information
            lowest_loss[i] = loss_full / len(valid_loader.dataset)   
            print("full VAE / AE emulator loss = {:0.3f}".format(lowest_loss[i]))

        if lowest_loss[i] < best_loss:
            best_loss = lowest_loss[i]
            torch.save(train_loss, save_dir+"train_loss.dat")
            torch.save(valid_loss, save_dir+"valid_loss.dat")
            torch.save(net.state_dict(), save_dir+'network-VAE.params')
            if structure_flag != 2:
                torch.save(train_loss_2, save_dir+"train_loss-latent.dat")
                torch.save(valid_loss_2, save_dir+"valid_loss-latent.dat")
                torch.save(net_latent.state_dict(), save_dir+'network-latent.params')

        # save after each iteration in case the job ends prematurely
        save_data = torch.vstack((lowest_loss, lowest_loss_2, lowest_loss_3))
        if vary_learning_rate == True: save_str="optimized-lr.dat"
        else: save_str = "optimized-bsize.dat"
        torch.save(save_data, save_dir+save_str)

    t2 = time.time()
    print("Done training networks!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))
    print("Lowest loss achieved was {:0.3f}".format(best_loss))

if __name__ == "__main__":
    main()
