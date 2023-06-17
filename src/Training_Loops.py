import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import src.Networks as Networks
import src.Dataset as Dataset

# ---------------------------------------------------------------------------
# Training Loops
# ---------------------------------------------------------------------------
def train_VAE(net, num_epochs, batch_size, beta, structure_flag,
              optimizer, train_loader, valid_loader, 
              print_progress = True, save_dir="", lr=0):
    """
    Train the VAE network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0
    net_save = Networks.Network_Emulator(structure_flag, True).to(Dataset.try_gpu())

    train_loss = torch.zeros([num_epochs], device=Dataset.try_gpu())
    valid_loss = torch.zeros([num_epochs], device=Dataset.try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
        train_KLD_sub = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; matrix = batch[1]
            prediction, mu, log_var = net(matrix.view(batch_size, 50, 50))

            loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
            assert torch.isnan(loss) == False 
            assert torch.isinf(loss) == False

            train_loss_sub += loss.item()
            train_KLD_sub += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()
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
            loss = Dataset.VAE_loss(prediction, matrix, mu, log_var, beta)
            valid_loss_sub += loss.item()
            valid_KLD_sub  += beta*(0.5 * torch.sum(log_var.exp() - log_var - 1 + mu.pow(2))).item()

        # Aggregate loss information
        train_loss[epoch] = train_loss_sub / len(train_loader.dataset)
        valid_loss[epoch] = valid_loss_sub / len(valid_loader.dataset)
        if valid_KLD_sub < 1e-7 and beta != 0:
            print("WARNING! KLD term is close to 0, indicating potential posterior collapse!")

        # save the network if the validation loss improved, else stop early if there hasn't been
        # improvement for several epochs
        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss.dat")
                torch.save(valid_loss, save_dir+"valid_loss.dat")
                torch.save(net.state_dict(), save_dir+'network-VAE.params')
            worse_epochs = 0
        else:
            worse_epochs+=1

        if print_progress == True:
            print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, train_loss[epoch], valid_loss[epoch], worse_epochs))
            if beta != 0: print("Avg train KLD: {:0.3f}, avg valid KLD: {:0.3f}".format(train_KLD_sub/len(train_loader.dataset), valid_KLD_sub/len(valid_loader.dataset)))

        if epoch > 15 and worse_epochs >= 15:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
            break
    print("initial lr {:0.5f}, bsize {:0.0f}: Best reconstruction validation loss was {:0.3f} after {:0.0f} epochs".format(lr, batch_size, best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss

def train_latent(net, num_epochs, optimizer, train_loader, valid_loader,
                 print_progress=True, save_dir=""):
    """
    Train the features network
    """
    best_loss = 1e10
    worse_epochs = 0
    net_save = Networks.Network_Latent()

    train_loss = torch.zeros([num_epochs])
    valid_loss = torch.zeros([num_epochs])
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        avg_train_loss = 0.
        for (i, batch) in enumerate(train_loader):
            params = batch[0]; features = batch[2]
            prediction = net(params)
            loss = Dataset.features_loss(prediction, features)
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
            loss = Dataset.features_loss(prediction, features)
            avg_valid_loss+= loss.item()

        # Aggregate loss information
        train_loss[epoch] = avg_train_loss / len(train_loader.dataset)
        valid_loss[epoch] = avg_valid_loss / len(valid_loader.dataset)

        if valid_loss[epoch] < best_loss:
            best_loss = valid_loss[epoch]
            net_save.load_state_dict(net.state_dict())
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss-latent.dat")
                torch.save(valid_loss, save_dir+"valid_loss-latent.dat")
                torch.save(net.state_dict(), save_dir+'network-latent.params')
            worse_epochs = 0
        else:
            worse_epochs+= 1
        if epoch % 10 == 0 and print_progress == True:
            print("Epoch : {:d}, avg train loss: {:0.3f}\t best validation loss: {:0.3f}".format(epoch, avg_train_loss / len(train_loader.dataset), best_loss))
        if epoch > 30 and worse_epochs >= 20:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs. Stopping...")
            break
    print("Latent net: Best validation loss was {:0.4f} after {:0.0f} epochs".format(best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss

def train_MLP(net, num_epochs, batch_size, structure_flag,
              optimizer, train_loader, valid_loader,
              print_progress=True, save_dir="", lr=0):
    """
    Train the pure MLP network
    """
    # Keep track of the best validation loss for early stopping
    best_loss = 1e10
    worse_epochs = 0
    net_save = Networks.Network_Emulator(structure_flag, True).to(Dataset.try_gpu())

    train_loss = torch.zeros([num_epochs], device=Dataset.try_gpu())
    valid_loss = torch.zeros([num_epochs], device=Dataset.try_gpu())
    for epoch in range(num_epochs):
        # Run through the training set and update weights
        net.train()
        train_loss_sub = 0.
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
            if save_dir != "":
                torch.save(train_loss, save_dir+"train_loss.dat")
                torch.save(valid_loss, save_dir+"valid_loss.dat")
                torch.save(net.state_dict(), save_dir+'network-VAE.params')
            worse_epochs = 0
        else:
            worse_epochs+=1

        if print_progress == True: print("Epoch : {:d}, avg train loss: {:0.3f}\t avg validation loss: {:0.3f}\t ({:0.0f})".format(epoch, train_loss[epoch], valid_loss[epoch], worse_epochs))

        if epoch > 15 and worse_epochs >= 15:
            if print_progress == True: print("Validation loss hasn't improved for", worse_epochs, "epochs, stopping...")
            break
    print("lr {:0.5f}, bsize {:0.0f}: Best validation loss was {:0.3f} after {:0.0f} epochs".format(lr, batch_size, best_loss, epoch - worse_epochs))
    return net_save, train_loss, valid_loss