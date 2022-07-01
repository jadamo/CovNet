import torch
import torch.nn as nn
from torch.nn import functional as F
import time, math

#sys.path.insert(0, '/home/joeadamo/Research/CovA-NN-Emulator')
from CovNet import Block_Decoder_Quad, Block_Encoder_Quad, Network_Features, Block_Encoder, Block_Decoder, \
                   MatrixDataset, Network_VAE_Quad, VAE_loss, features_loss, try_gpu

# Total number of matrices in the training + validation + test set
N = 52500
#N = 20000

# wether to train using the percision matrix instead - NOT YET IMPLIMENTED
train_inverse = False
# wether or not to train with the correlation matrix + diagonal
train_correlation = False
# wether to train using the log of the matrix
train_log = True
# wether or not to train with the Cholesky decomposition
train_cholesky = False
# wether to train the VAE and features nets
do_VAE = True; do_features = True

training_dir = "/home/joeadamo/Research/Data/Training-Set/"
save_dir = "/home/joeadamo/Research/CovA-NN-Emulator/Data/Quad-decomp/"

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

def train_VAE(net_q1, net_q2, net_q3, num_epochs, batch_size, 
              optimizer_q1, optimizer_q2, optimizer_q3, train_loader, valid_loader):
    """
    Train all 3 VAE networks - one for each quadrant
    """
    loss_data = torch.zeros([6, num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net_q1.train(); net_q2.train(); net_q3.train()
        avg_train_loss_q1, avg_train_loss_q2, avg_train_loss_q3 = 0., 0., 0.
        avg_train_KLD = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1].view(batch_size, 100, 100)
            prediction_q1, mu_q1, log_var_q1 = net_q1(matrix[:,:50,:50])
            prediction_q2, mu_q2, log_var_q2 = net_q2(matrix[:,50:,50:])
            prediction_q3, mu_q3, log_var_q3 = net_q3(matrix[:,50:,:50])

            loss_q1 = VAE_loss(prediction_q1, matrix[:,:50,:50], mu_q1, log_var_q1, BETA)
            loss_q2 = VAE_loss(prediction_q2, matrix[:,50:,50:], mu_q2, log_var_q2, BETA)
            loss_q3 = VAE_loss(prediction_q3, matrix[:,50:,:50], mu_q3, log_var_q3, BETA)

            assert torch.isnan(loss_q1) == False and torch.isinf(loss_q1) == False

            avg_train_loss_q1 += loss_q1.item()
            avg_train_loss_q2 += loss_q2.item()
            avg_train_loss_q3 += loss_q3.item()
            avg_train_KLD += BETA*(0.5 * torch.sum(log_var_q1.exp() - log_var_q1 - 1 + mu_q1.pow(2))).item()
            
            optimizer_q1.zero_grad(); optimizer_q2.zero_grad(); optimizer_q3.zero_grad()
            loss_q1.backward(); loss_q2.backward(); loss_q3.backward()  
            optimizer_q1.step(); optimizer_q2.step(); optimizer_q3.step()

        # run through the validation set
        net_q1.eval(); net_q2.eval(); net_q3.eval()
        avg_valid_loss_q1, avg_valid_loss_q2, avg_valid_loss_q3 = 0., 0., 0.
        avg_valid_KLD = 0.
        for params, matrix in valid_loader:
            prediction_q1, mu_q1, log_var_q1 = net_q1(matrix[:,:50,:50])
            prediction_q2, mu_q2, log_var_q2 = net_q2(matrix[:,50:,50:])
            prediction_q3, mu_q3, log_var_q3 = net_q3(matrix[:,50:,:50])
            #prediction = prediction.view(batch_size, 100, 100)
            loss_q1 = VAE_loss(prediction_q1, matrix[:,:50,:50], mu_q1, log_var_q1, BETA)
            loss_q2 = VAE_loss(prediction_q2, matrix[:,50:,50:], mu_q2, log_var_q2, BETA)
            loss_q3 = VAE_loss(prediction_q3, matrix[:,50:,:50], mu_q3, log_var_q3, BETA)
            avg_valid_loss_q1+= loss_q1.item()
            avg_valid_loss_q2+= loss_q2.item()
            avg_valid_loss_q3+= loss_q3.item()
            avg_valid_KLD += BETA*(0.5 * torch.sum(log_var_q1.exp() - log_var_q1 - 1 + mu_q1.pow(2))).item()

        # Aggregate loss information
        loss_data[0, epoch] = avg_train_loss_q1 / len(train_loader.dataset)
        loss_data[1, epoch] = avg_train_loss_q2 / len(train_loader.dataset)
        loss_data[2, epoch] = avg_train_loss_q3 / len(train_loader.dataset)
        loss_data[3, epoch] = avg_valid_loss_q1 / len(valid_loader.dataset)
        loss_data[4, epoch] = avg_valid_loss_q2 / len(valid_loader.dataset)
        loss_data[5, epoch] = avg_valid_loss_q3 / len(valid_loader.dataset)
        print("Epoch : {:d}, avg train loss: {:0.3f}, {:0.3f}, {:0.3f}".format(epoch, loss_data[0,epoch], loss_data[1,epoch], loss_data[2,epoch]))
        print("Avg validation loss: {:0.3f}, {:0.3f}, {:0.3f}".format(loss_data[3,epoch], loss_data[4,epoch], loss_data[5,epoch]))
        print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(avg_train_KLD/len(train_loader.dataset), avg_valid_KLD/len(valid_loader.dataset)))
        if avg_valid_KLD < 1e-7:
            print("WARNING! KLD term is close to 0, indicating potential posterior collapse!")
    return net_q1, net_q2, net_q3, loss_data

def train_features(net_f1, net_f2, net_f3, num_epochs, 
                   optimizer_f1, optimizer_f2, optimizer_f3, train_loader, valid_loader):
    """
    Train the features network
    """

    loss_data = torch.zeros([6, num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net_f1.train(); net_f2.train(); net_f3.train()
        avg_train_loss_f1, avg_train_loss_f2, avg_train_loss_f3 = 0., 0., 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction_f1 = net_f1(params)
            prediction_f2 = net_f2(params)
            prediction_f3 = net_f3(params)
            loss_f1 = features_loss(prediction_f1, features[:,0])
            loss_f2 = features_loss(prediction_f2, features[:,1])
            loss_f3 = features_loss(prediction_f3, features[:,2])
            assert torch.isnan(loss_f1) == False and torch.isinf(loss_f1) == False

            avg_train_loss_f1 += loss_f1.item()
            avg_train_loss_f2 += loss_f2.item()
            avg_train_loss_f3 += loss_f3.item()
            optimizer_f1.zero_grad(); optimizer_f2.zero_grad(); optimizer_f3.zero_grad()
            loss_f1.backward(); loss_f2.backward(); loss_f3.backward()
            optimizer_f1.step(); optimizer_f2.step(); optimizer_f3.step()

        # run through the validation set
        net_f1.eval(); net_f2.eval(); net_f3.eval()
        avg_valid_loss_f1, avg_valid_loss_f2, avg_valid_loss_f3 = 0., 0., 0.
        for params, matrix, features in valid_loader:
            prediction_f1 = net_f1(params)
            prediction_f2 = net_f2(params)
            prediction_f3 = net_f3(params)
            loss_f1 = features_loss(prediction_f1, features[:,0])
            loss_f2 = features_loss(prediction_f2, features[:,1])
            loss_f3 = features_loss(prediction_f3, features[:,2])
            avg_valid_loss_f1 += loss_f1.item()
            avg_valid_loss_f2 += loss_f2.item()
            avg_valid_loss_f3 += loss_f3.item()

        # Aggregate loss information
        loss_data[0, epoch] = avg_train_loss_f1 / len(train_loader.dataset)
        loss_data[1, epoch] = avg_train_loss_f2 / len(train_loader.dataset)
        loss_data[2, epoch] = avg_train_loss_f3 / len(train_loader.dataset)
        loss_data[3, epoch] = avg_valid_loss_f1 / len(valid_loader.dataset)
        loss_data[4, epoch] = avg_valid_loss_f2 / len(valid_loader.dataset)
        loss_data[5, epoch] = avg_valid_loss_f3 / len(valid_loader.dataset)
    return net_f1, net_f2, net_f3, loss_data

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
    num_epochs = 75
    num_epochs_2 = 130

    N_train = int(N*0.8)
    N_valid = int(N*0.1)

    # initialize network
    net_q1 = Network_VAE_Quad(50, True, train_cholesky).to(try_gpu())
    net_q2 = Network_VAE_Quad(50, True, train_cholesky).to(try_gpu())
    net_q3 = Network_VAE_Quad(50, False, train_cholesky).to(try_gpu())
    net_f1 = Network_Features(6, 10).to(try_gpu())
    net_f2 = Network_Features(6, 10).to(try_gpu())
    net_f3 = Network_Features(6, 10).to(try_gpu())

    net_q1.apply(He)
    net_q2.apply(He)
    net_q3.apply(He)
    net_f1.apply(xavier)
    net_f2.apply(xavier)
    net_f3.apply(xavier)

    # Define the optimizer
    optimizer_q1 = torch.optim.Adam(net_q1.parameters(), lr=lr)
    optimizer_q2 = torch.optim.Adam(net_q2.parameters(), lr=lr)
    optimizer_q3 = torch.optim.Adam(net_q3.parameters(), lr=lr)
    optimizer_f1 = torch.optim.Adam(net_f1.parameters(), lr=lr_2)
    optimizer_f2 = torch.optim.Adam(net_f2.parameters(), lr=lr_2)
    optimizer_f3 = torch.optim.Adam(net_f3.parameters(), lr=lr_2)

    # get the training / test datasets
    t1 = time.time()
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
        net_q1, net_q2, net_q3, loss_data = train_VAE(net_q1, net_q2, net_q3, num_epochs, batch_size, optimizer_q1, optimizer_q2, optimizer_q3, train_loader, valid_loader)
        t2 = time.time()
        print("Done training VAE!, took {:0.0f} minutes {:0.2f} seconds\n".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

        # Save the network and loss data to disk
        torch.save(loss_data, save_dir+"loss_data.dat")
        torch.save(net_q1.state_dict(), save_dir+'network-VAE-q1.params')
        torch.save(net_q2.state_dict(), save_dir+'network-VAE-q2.params')
        torch.save(net_q3.state_dict(), save_dir+'network-VAE-q3.params')

    # next, train the secondary network with the features from the VAE as the output
    if do_features:
        # If train_net is false, assume we already trained the VAE net and load it in
        if do_VAE == False:
            net_q1.load_state_dict(torch.load(save_dir+'network-VAE-q1.params'))
            net_q2.load_state_dict(torch.load(save_dir+'network-VAE-q2.params'))
            net_q3.load_state_dict(torch.load(save_dir+'network-VAE-q3.params'))

        # separate encoder and decoders
        encoder_q1 = Block_Encoder_Quad(50, True).to(try_gpu())
        decoder_q1 = Block_Decoder_Quad(50, True, train_cholesky).to(try_gpu())
        encoder_q1.load_state_dict(net_q1.Encoder.state_dict())
        decoder_q1.load_state_dict(net_q1.Decoder.state_dict())

        encoder_q2 = Block_Encoder_Quad(50, True).to(try_gpu())
        decoder_q2 = Block_Decoder_Quad(50, True, train_cholesky).to(try_gpu())
        encoder_q2.load_state_dict(net_q2.Encoder.state_dict())
        decoder_q2.load_state_dict(net_q2.Decoder.state_dict())

        encoder_q3 = Block_Encoder_Quad(50, False).to(try_gpu())
        decoder_q3 = Block_Decoder_Quad(50, False, train_cholesky).to(try_gpu())
        encoder_q3.load_state_dict(net_q3.Encoder.state_dict())
        decoder_q3.load_state_dict(net_q3.Decoder.state_dict())

        # gather feature data by running thru the trained encoder
        train_f = torch.zeros(N_train, 3, 10, device=try_gpu())
        valid_f = torch.zeros(N_valid, 3, 10, device=try_gpu())
        encoder_q1.eval(); encoder_q2.eval(); encoder_q3.eval()
        for n in range(N_train):
            matrix = train_data[n][1].view(1,100,100)
            z_q1, mu, log_var = encoder_q1(matrix[:,:50,:50])
            z_q2, mu, log_var = encoder_q2(matrix[:,50:,50:])
            z_q3, mu, log_var = encoder_q3(matrix[:,50:,:50])
            train_f[n, 0] = z_q1.view(10)
            train_f[n, 1] = z_q2.view(10)
            train_f[n, 2] = z_q3.view(10)
        for n in range(N_valid):
            matrix = valid_data[n][1].view(1,100,100)
            z_q1, mu, log_var = encoder_q1(matrix[:,:50,:50])
            z_q2, mu, log_var = encoder_q2(matrix[:,50:,50:])
            z_q3, mu, log_var = encoder_q3(matrix[:,50:,:50])
            valid_f[n, 0] = z_q1.view(10)
            valid_f[n, 1] = z_q2.view(10)
            valid_f[n, 2] = z_q3.view(10)

        # add feature data to the training set and reinitialize the data loaders
        train_data.add_features(train_f)
        valid_data.add_features(valid_f)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

        # train the secondary network!
        t1 = time.time()
        net_f1, net_f2, net_f3, loss_data = train_features(net_f1, net_f2, net_f3, num_epochs_2, optimizer_f1, optimizer_f2, optimizer_f3, train_loader, valid_loader)
        t2 = time.time()
        print("Done training feature net!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

        torch.save(loss_data, save_dir+"loss_data-features.dat")
        torch.save(net_f1.state_dict(), save_dir+'network-latent-q1.params')
        torch.save(net_f2.state_dict(), save_dir+'network-latent-q2.params')
        torch.save(net_f3.state_dict(), save_dir+'network-latent-q3.params')

if __name__ == "__main__":
    main()
