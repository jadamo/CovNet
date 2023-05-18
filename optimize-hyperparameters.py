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
# 0 = fully-connected ResNet
# 1 = CNN ResNet
# 2 = simple (no VAE, just a simple fully connected network)
structure_flag = 2

training_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
#training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-HighZ-NGC/"

if structure_flag == 0: folder = "full"
elif structure_flag == 1: folder = "cnn"
elif structure_flag == 2: folder = "simple"
folder+="/"

save_dir = "/home/u12/jadamo/CovNet/emulators/ngc_z3/"+folder
#save_dir = "/home/joeadamo/Research/CovNet/emulators/ngc_z3/"+folder

# parameter to control the importance of the KL divergence loss term
# A large value might result in posterior collapse
BETA = 0.01

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


def train_VAE(net, lr, bsize, num_epochs, batch_size, optimizer, train_loader, valid_loader):
    """
    Train the VAE network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0
    net_save = CovNet.Network_VAE(structure_flag, True).to(CovNet.try_gpu())

    train_loss = torch.zeros([num_epochs], device=CovNet.try_gpu())
    valid_loss = torch.zeros([num_epochs], device=CovNet.try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
        train_KLD_sub = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 50, 50))
            #prediction = prediction.view(batch_size, 100, 100)
            #print(torch.min(prediction), torch.max(prediction))
            loss = CovNet.VAE_loss(prediction, matrix, mu, log_var, BETA)
            try:
                assert torch.isnan(loss) == False 
                assert torch.isinf(loss) == False
            except:
                print("loss is nan or infinity! breaking early...")
                break

            train_loss_sub += loss.item()
            train_KLD_sub += BETA*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e8)    
            optimizer.step()

        # run through the validation set
        net.eval()
        valid_loss_sub = 0.
        valid_KLD_sub = 0.
        for (i, batch) in enumerate(valid_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 50, 50))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = CovNet.VAE_loss(prediction, matrix, mu, log_var, BETA)
            valid_loss_sub += loss.item()
            valid_KLD_sub  += BETA*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()

        # Aggregate loss information
        train_loss[epoch] = train_loss_sub / len(train_loader.dataset)
        valid_loss[epoch] = valid_loss_sub / len(valid_loader.dataset)
        if valid_KLD_sub < 1e-7:
            print("WARNING! KLD term is close to 0, indicating potential posterior collapse!")

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for several epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            worse_epochs = 0
        else:
            worse_epochs+=1

        if epoch > 15 and worse_epochs >= 15:
            break
    print("lr {:0.5f}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(lr, bsize, best_loss, epoch - worse_epochs))
    return best_loss, train_loss, valid_loss, net_save

def train_latent(net, num_epochs, optimizer, train_loader, valid_loader):
    """
    Train the features network
    """
    best_loss = 1e10
    worse_epochs = 0
    net_save = CovNet.Network_Latent(False)

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction = net(params)
            loss = CovNet.features_loss(prediction, features)
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
            loss = CovNet.features_loss(prediction, features)
            avg_valid_loss+= loss.item()

        # Aggregate loss information
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)

        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            worse_epochs = 0
        else:
            worse_epochs+= 1
        if epoch > 30 and worse_epochs >= 20:
            print("Validation loss hasn't improved for", worse_epochs, "epochs. Stopping...")
            break
    print("Best latent net validation loss was {:0.4f} after {:0.0f} epochs".format(best_loss, epoch - worse_epochs))
    return best_loss, train_loss, valid_loss, net

def train_simple(net, lr, bsize, num_epochs, batch_size, optimizer, train_loader, valid_loader):
    """
    Train the VAE network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0

    net_save = CovNet.Network_VAE(structure_flag, True).to(CovNet.try_gpu())

    train_loss = torch.zeros([num_epochs], device=CovNet.try_gpu())
    valid_loss = torch.zeros([num_epochs], device=CovNet.try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
        train_KLD_sub = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params.view(batch_size, 6))

            loss = F.l1_loss(prediction, matrix, reduction="sum")
            assert torch.isnan(loss) == False 
            assert torch.isinf(loss) == False

            train_loss_sub += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e8)    
            optimizer.step()

        # run through the validation set
        net.eval()
        valid_loss_sub = 0.
        valid_KLD_sub = 0.
        for (i, batch) in enumerate(valid_loader):
            params = batch[0]; matrix = batch[1]
            prediction = net(params.view(batch_size, 6))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = F.l1_loss(prediction, matrix, reduction="sum")
            valid_loss_sub += loss.item()

        # Aggregate loss information
        train_loss[epoch] = train_loss_sub / len(train_loader.dataset)
        valid_loss[epoch] = valid_loss_sub / len(valid_loader.dataset)

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for several epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            worse_epochs = 0
        else:
            worse_epochs+=1

        if epoch > 15 and worse_epochs >= 15:
            break
    print("lr {:0.5f}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(lr, bsize, best_loss, epoch - worse_epochs))
    return best_loss, train_loss, valid_loss, net_save

def main():

    print("Optimizing learning rate:", vary_learning_rate)
    print("Optimizing batch size:", vary_batch_size)
    print("Saving to", save_dir)
    print("Using GPU:", torch.cuda.is_available())
    print("network structure flag =", structure_flag)

    lr_VAE    = 0.0025
    if vary_learning_rate == True: lr_VAE = torch.logspace(-4, -2, 20).to(CovNet.try_gpu())
    else: lr_VAE = 0.0025

    if vary_batch_size == True: batch_size = torch.Tensor([25, 50, 100, 200, 265, 424, 530])
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
    best_loss = 1e5

    t1 = time.time()
    for i in range(iterate):

        # re-initialize networks
        net = CovNet.Network_VAE(structure_flag, True).to(CovNet.try_gpu())
        net_latent = CovNet.Network_Latent(False)

        net.apply(He)
        net_latent.apply(xavier)

        lr = lr_VAE[i] if vary_learning_rate == True else lr_VAE
        bsize = batch_size[i] if vary_batch_size == True else batch_size

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=bsize, shuffle=True)

        # Define the optimizer
        optimizer_VAE = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer_latent = torch.optim.Adam(net_latent.parameters(), lr=lr_latent)
        
        # Train the network!
        if structure_flag != 2: 
            lowest_loss[i], train_loss, valid_loss, net = train_VAE(net, lr, bsize, num_epochs_VAE, batch_size, optimizer_VAE, train_loader, valid_loader)
        else: 
            lowest_loss[i], train_loss, valid_loss, net = train_simple(net, lr, bsize, num_epochs_VAE, batch_size, optimizer_VAE, train_loader, valid_loader)

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

            # train the secondary network!
            lowest_loss_2[i], train_loss_2, valid_loss_2, net_latent = train_latent(net_latent, num_epochs_latent, optimizer_latent, train_loader, valid_loader)
        
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
        save_data = torch.vstack((lowest_loss, lowest_loss_2))
        if vary_learning_rate == True: save_str="optimized-lr.dat"
        else: save_str = "optimized-bsize.dat"
        torch.save(save_data, save_dir+save_str)

    t2 = time.time()
    print("Done training networks!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))
    print("Lowest loss achieved was {:0.3f}".format(best_loss))

if __name__ == "__main__":
    main()
