import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
from CovNet import Network_Features, Block_Encoder, Block_Decoder, \
                   Network_VAE, MatrixDataset, VAE_loss, features_loss, try_gpu

# Total number of matrices in the training + validation + test set
N = 52500
#N = 5000

# wether to train using the percision matrix instead - NOT YET IMPLIMENTED
train_inverse = False
# wether or not to train with the correlation matrix + diagonal
train_correlation = True
# wether to train using the log of the matrix
train_log = True
# wether or not to train with the Cholesky decomposition
train_cholesky = False
# wether to train the VAE and features nets
do_VAE = True; do_features = True

# beta to control the importance of the KL divergence loss term
BETA = 30

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

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        avg_train_KLD = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 100, 100))
            #prediction = prediction.view(batch_size, 100, 100)
            #print(torch.min(prediction), torch.max(prediction))
            loss = VAE_loss(prediction, matrix, mu, log_var, BETA)
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
        min_val = 1e30; max_val = -1e10
        min_pre = 1e30; max_pre = -1e10
        for params, matrix in valid_loader:
            prediction, mu, log_var = net(matrix.view(batch_size, 100, 100))
            #prediction = prediction.view(batch_size, 100, 100)
            loss = VAE_loss(prediction, matrix, mu, log_var, BETA)
            avg_valid_loss+= loss.item()
            avg_valid_KLD += BETA*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
            #min_pre = min(torch.min(prediction), min_pre)
            #max_pre = max(torch.max(prediction), max_pre)
            #min_val = min(torch.min(matrix), min_val)
            #max_val = max(torch.max(matrix), max_val)

        # Aggregate loss information
        print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}".format(epoch, avg_train_loss / len(train_loader.dataset), avg_valid_loss / len(valid_loader.dataset)))
        print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(avg_train_KLD/len(train_loader.dataset), avg_valid_KLD/len(valid_loader.dataset)))
        #print(" min valid = {:0.3f}, max valid = {:0.3f}".format(min_val, max_val))
        #print(" min predict = {:0.3f}, max predict = {:0.3f}".format(min_pre, max_pre))
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)
    return net, train_loss, valid_loss

def train_features(net, num_epochs, optimizer, train_loader, valid_loader):
    """
    Train the features network
    """

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction = net(params)
            loss = features_loss(prediction, features)
            assert torch.isnan(loss) == False and torch.isinf(loss) == False

            avg_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # run through the validation set
        net.eval()
        avg_valid_loss = 0.
        min_val = 1e30; max_val = -1e10
        min_pre = 1e30; max_pre = -1e10
        for params, matrix, features in valid_loader:
            prediction = net(params)
            loss = features_loss(prediction, features)
            avg_valid_loss+= loss.item()

        # Aggregate loss information
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)
    return net, train_loss, valid_loss

def main():

    #print("Training with inverse matrices:       " + str(train_inverse))
    print("Training with correlation matrices:   " + str(train_correlation))
    print("Training with log matrices:           " + str(train_log))
    print("Training with cholesky decomposition: " + str(train_cholesky))
    print("Training VAE net: features net:      [" + str(do_VAE) + ", " + str(do_features) + "]")

    assert not (train_correlation == True and train_cholesky == True), "Cannot train with correlation and cholesky decompositions simultaneously"

    batch_size = 50
    lr = 0.003
    lr_2 = 0.008
    num_epochs = 60
    num_epochs_2 = 130

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    net = Network_VAE().to(try_gpu())
    net_2 = Network_Features(6, 10).to(try_gpu())

    net.apply(He)
    net_2.apply(xavier)

    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer_2 = torch.optim.Adam(net_2.parameters(), lr=lr_2)

    # get the training / test datasets
    t1 = time.time()
    training_dir = "/home/joeadamo/Research//Data/Training-Set/"
    save_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Data/"
    train_data = MatrixDataset(training_dir, N_train, 0, train_log, train_correlation, train_cholesky)
    valid_data = MatrixDataset(training_dir, N_valid, N_train, train_log, train_correlation, train_cholesky)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    t2 = time.time()
    print("Done loading in data, took {:0.2f} s".format(t2 - t1))
    #train_in, train_out, test_in, test_out = read_training_set(training_dir, 0.8)

    # Train the network!
    if do_VAE:
        t1 = time.time()
        net, train_loss, valid_loss = train_VAE(net, num_epochs, batch_size, optimizer, train_loader, valid_loader)
        t2 = time.time()
        print("Done training VAE!, took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

        # Save the network and loss data to disk
        torch.save(train_loss, save_dir+"train_loss.dat")
        torch.save(valid_loss, save_dir+"valid_loss.dat")
        torch.save(net.state_dict(), save_dir+'network-VAE.params')

    # next, train the secondary network with the features from the VAE as the output
    if do_features:
        # If train_net is false, assume we already trained the VAE net and load it in
        if do_VAE == False:
            net = Network_VAE()
            net.load_state_dict(torch.load(save_dir+'network-VAE.params'))

        # separate encoder and decoders
        encoder = Block_Encoder().to(try_gpu())
        decoder = Block_Decoder().to(try_gpu())
        encoder.load_state_dict(net.Encoder.state_dict())
        decoder.load_state_dict(net.Decoder.state_dict())

        # gather feature data by running thru the trained encoder
        train_f = torch.zeros(N_train, 10, device=try_gpu())
        valid_f = torch.zeros(N_valid, 10, device=try_gpu())
        encoder.eval()
        for n in range(N_train):
            matrix = train_data[n][1].view(1,100,100)
            z, mu, log_var = encoder(matrix)
            train_f[n] = z.view(10)
        for n in range(N_valid):
            matrix = valid_data[n][1].view(1,100,100)
            z, mu, log_var = encoder(matrix)
            valid_f[n] = z.view(10)

        # add feature data to the training set and reinitialize the data loaders
        train_data.add_features(train_f)
        valid_data.add_features(valid_f)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

        # train the secondary network!
        t1 = time.time()
        net_2, train_loss, valid_loss = train_features(net_2, num_epochs_2, optimizer_2, train_loader, valid_loader)
        t2 = time.time()
        print("Done training feature net!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

        torch.save(train_loss, save_dir+"train_loss-features.dat")
        torch.save(valid_loss, save_dir+"valid_loss-features.dat")
        torch.save(net_2.state_dict(), save_dir+'network-features.params')

if __name__ == "__main__":
    main()
