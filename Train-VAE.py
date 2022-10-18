import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
import CovNet

# Total number of matrices in the training + validation + test set
N = 65500
#N = 20000

# wether to train using the log of the matrix
train_log = True
# wether or not to train on just the gaussian covariance (this is a test)
train_gaussian = False
# wether or not to train with the Cholesky decomposition
train_cholesky = False
# wether to train the VAE and features nets
do_VAE = True; do_features = True

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-HighZ-NGC/"
if train_gaussian == True:   save_dir = "/home/joeadamo/Research/CovNet/Data/Gaussian/"
elif train_cholesky == True: save_dir = "/home/joeadamo/Research/CovNet/Data/Cholesky-decomp/"
else: save_dir = "/home/joeadamo/Research/CovNet/Data/Non-Gaussian/"

# parameter to control the importance of the KL divergence loss term
# A large value might result in posterior collapse
BETA = 0.005

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

def train_VAE(net, num_epochs, batch_size, optimizer, train_loader, valid_loader):
    """
    Train the VAE network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        avg_train_KLD = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 80, 80))
            #prediction = prediction.view(batch_size, 100, 100)
            #print(torch.min(prediction), torch.max(prediction))
            loss = CovNet.VAE_loss(prediction, matrix, mu, log_var, BETA)
            assert torch.isnan(loss) == False and torch.isinf(loss) == False

            avg_train_loss += loss.item()
            avg_train_KLD += BETA*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1e8)    
            optimizer.step()

        # run through the validation set
        net.eval()
        avg_valid_loss = 0.
        avg_valid_KLD = 0.
        for (i, batch) in enumerate(valid_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 80, 80))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = CovNet.VAE_loss(prediction, matrix, mu, log_var, BETA)
            avg_valid_loss+= loss.item()
            avg_valid_KLD += BETA*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()

        # Aggregate loss information
        print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}".format(epoch, avg_train_loss / len(train_loader.dataset), avg_valid_loss / len(valid_loader.dataset)))
        print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(avg_train_KLD/len(train_loader.dataset), avg_valid_KLD/len(valid_loader.dataset)))
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)
        if avg_valid_KLD < 1e-7:
            print("WARNING! KLD term is close to 0, indicating potential posterior collapse!")

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for 5 epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            torch.save(train_loss, save_dir+"train_loss.dat")
            torch.save(valid_loss, save_dir+"valid_loss.dat")
            torch.save(net.state_dict(), save_dir+'network-VAE.params')
            worse_epochs = 0
        else:
            worse_epochs+=1
        if epoch > 15 and worse_epochs >= 12:
            print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
            break

    return

def train_latent(net, num_epochs, optimizer, train_loader, valid_loader):
    """
    Train the features network
    """
    best_loss = 1e10
    worse_epochs = 0

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
            torch.save(train_loss, save_dir+"train_loss-latent.dat")
            torch.save(valid_loss, save_dir+"valid_loss-latent.dat")
            torch.save(net.state_dict(), save_dir+'network-latent.params')
            worse_epochs = 0
        else:
            worse_epochs+= 1
        if epoch > 30 and worse_epochs >= 20:
            print("Validation loss hasn't improved for", worse_epochs, "epochs. Stopping...")
            break

def main():

    #print("Training with inverse matrices:       " + str(train_inverse))
    #print("Training with correlation matrices:   " + str(train_correlation))
    print("Training with log matrices:           " + str(train_log))
    print("Training with cholesky decomposition: " + str(train_cholesky))
    print("Training with just gaussian term:     " + str(train_gaussian))
    print("Training VAE net: features net:      [" + str(do_VAE) + ", " + str(do_features) + "]")

    batch_size = 50
    lr = 0.0025
    lr_2 = 0.008
    num_epochs = 70
    num_epochs_2 = 200

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    net = CovNet.Network_VAE(train_cholesky).to(CovNet.try_gpu())
    net_2 = CovNet.Network_Features(7, 10).to(CovNet.try_gpu())

    net.apply(He)
    net_2.apply(xavier)

    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(net_2.parameters(), lr=lr_2)

    # get the training / test datasets
    t1 = time.time()
    train_data = CovNet.MatrixDataset(training_dir, N_train, 0, train_log, train_gaussian, train_cholesky)
    valid_data = CovNet.MatrixDataset(training_dir, N_valid, N_train, train_log, train_gaussian, train_cholesky)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))

    # Train the network! Progress is saved to file within the function
    if do_VAE:
        t1 = time.time()
        train_VAE(net, num_epochs, batch_size, optimizer, train_loader, valid_loader)
        t2 = time.time()
        print("Done training VAE!, took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    # next, train the secondary network with the features from the VAE as the output
    if do_features:

        # In case the network went thru early stopping, reload the net that was saved to file
        net.load_state_dict(torch.load(save_dir+'network-VAE.params'))
        # separate encoder and decoders
        encoder = CovNet.Block_Encoder().to(CovNet.try_gpu())
        decoder = CovNet.Block_Decoder(train_cholesky).to(CovNet.try_gpu())
        encoder.load_state_dict(net.Encoder.state_dict())
        decoder.load_state_dict(net.Decoder.state_dict())

        # gather feature data by running thru the trained encoder
        train_f = torch.zeros(N_train, 10, device=CovNet.try_gpu())
        valid_f = torch.zeros(N_valid, 10, device=CovNet.try_gpu())
        encoder.eval()
        for n in range(N_train):
            matrix = train_data[n][1].view(1,80,80)
            z, mu, log_var = encoder(matrix)
            train_f[n] = z.view(10)
        for n in range(N_valid):
            matrix = valid_data[n][1].view(1,80,80)
            z, mu, log_var = encoder(matrix)
            valid_f[n] = z.view(10)

        # add feature data to the training set and reinitialize the data loaders
        train_data.add_features(train_f)
        valid_data.add_features(valid_f)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

        # train the secondary network!
        t1 = time.time()
        train_latent(net_2, num_epochs_2, optimizer_2, train_loader, valid_loader)
        t2 = time.time()
        print("Done training feature net!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()
