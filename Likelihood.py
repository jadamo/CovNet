import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
from tqdm import tqdm
from classy import Class
import io
from contextlib import redirect_stdout

sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet
import tracemalloc

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-NG/"
data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"

use_T0 = True

C_fid_file = np.load(data_dir+"Cov_Fid.npz")
if use_T0 == True:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"] + C_fid_file["C_T0"]
else:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"]

P_fid = np.linalg.inv(C_fid)

cosmo_prior = np.array([[66.0, 75.5],
                        [0.10782, 0.13178],
                        [0.0211375, 0.0233625],
                        [2.4752, 3.7128],#[1.1885e-9, 2.031e-9],
                        [1.9, 2.45],
                        [-3.562, 0.551]])

#                     H0,omch2, ombh2,  As,  b1,     b2
cosmo_fid = np.array([67.8,0.1190,0.02215,3.094,1.9485,-0.5387])

gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}
redshift = 0.58

def PCA_emulator():
    """
    Sets up the PCA covariance matrix emulator to use in likelihood analysis
    """
    N_C = 100
    C_PCA = np.zeros((N_C, 100, 100))
    params_PCA = np.zeros((N_C, 6))
    for i in range(N_C):
        temp = np.load(PCA_dir+"CovNG-"+f'{i:04d}'+".npz")
        params_PCA[i] = temp["params"]
        C_PCA[i] = temp["C"]
    
        # if the matrix doesn't match the transpose close enough, manually flip over the diagonal
        try:
            np.testing.assert_allclose(C_PCA[i], C_PCA[i].T, err_msg="covariance must match transpose")
        except AssertionError:
            L = np.tril(C_PCA[i])
            U = np.tril(C_PCA[i], k=-1).T
            C_PCA[i] = L + U
    try:
        Emu = ce.CovEmu(params_PCA, C_PCA, NPC_D=20, NPC_L=20)
    except np.linalg.LinAlgError:
        print("WARNING! PCA emulator unable to initialize - input matrices aren't positive definite!")
        Emu = None
    return Emu

def CovNet_emulator():
    """
    Returns a neural net covariance matrix emulator
    """
    VAE = CovNet.Network_VAE().to(CovNet.try_gpu());       VAE.eval()
    decoder = CovNet.Block_Decoder().to(CovNet.try_gpu()); decoder.eval()
    net = CovNet.Network_Features(6, 10).to(CovNet.try_gpu())
    VAE.load_state_dict(torch.load(data_dir+'Non-Gaussian/network-VAE.params', map_location=CovNet.try_gpu()))
    decoder.load_state_dict(VAE.Decoder.state_dict())
    net.load_state_dict(torch.load(data_dir+"Non-Gaussian/network-features.params", map_location=CovNet.try_gpu()))

    return decoder, net

def get_covariance(decoder, net, theta):
    # first convert theta to the format expected by our emulators
    params = torch.tensor([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]]).float()
    features = net(params); C = decoder(features.view(1,10)).view(100, 100)
    C = CovNet.corr_to_cov(C).cpu().detach().numpy()
    return C

# def get_covariance(Emu, theta):
#     # first convert theta to the format expected by our emulators
#     params = np.array([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]])
#     C = Emu.predict(params)
#     return C

def sample_prior(cosmo_prior):
    """
    Draws a random starting point based on the prior and returns it
    """
    nParams = cosmo_prior.shape[0] # the number of parameters
    theta0 = np.zeros((nParams))
    for i in range(nParams):
        theta0[i] = (cosmo_prior[i,1]-cosmo_prior[i,0])* np.random.rand(1) + cosmo_prior[i,0] # randomly choose a value in the acceptable range
    
    return theta0 

def model_vector(params, gparams, pgg):
    """
    Calculates the model vector using Yosuke's galaxy power spectrum emulator
    """
    #print(params)
    h = params[0] / 100
    omch2 = params[1]
    ombh2 = params[2]
    #assert omch2 <= 0.131780
    As = params[3]
    #assert As >= 2.47520
    ns = 0.965
    Om0 = (omch2 + ombh2 + 0.00064) / (h**2)
    
    # rebuild parameters into correct format (ombh2, omch2, 1-Om0, ln As, ns, w)
    cparams = np.array([ombh2, omch2, 1-Om0, As, ns, -1])
    redshift = 0.5
    k = np.linspace(0.005, 0.25, 50)
    mu = np.linspace(0.1,0.9,4)
    alpha_perp = 1.1
    alpha_para = 1

    pgg.set_cosmology(cparams, redshift) # <- takes ~0.17s to run
    pgg.set_galaxy(gparams)
    # takes ~0.28 s to run
    P0_emu = pgg.get_pl_gg_ref(0, k, alpha_perp, alpha_para, name='total')
    P2_emu = pgg.get_pl_gg_ref(2, k, alpha_perp, alpha_para, name='total')
    return np.concatenate((P0_emu, P2_emu))

def model_vector_CLASS_PT(params):
    z = 0.58
    cosmo = Class()
    cosmo.set({'output':'mPk',
            'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes', # use CDM+baryon spectra
            'RSD':'Yes',
            'AP':'Yes', # Alcock-Paczynski effect
            'Omfid':'0.31', # fiducial Omega_m
            'PNG':'No', # single-field inflation PNG
            'FFTLog mode':'FAST',
            'A_s':np.exp(params[3])/1e10,
            'n_s':0.9649,
            'tau_reio':0.052,
            'omega_b':params[2],
            'omega_cdm':params[1],
            'h':params[0] / 100.,
            'YHe':0.2425,
            'N_ur':2.0328,
            'N_ncdm':1,
            'm_ncdm':0.06,
            'z_pk':z
            })  
    k = np.linspace(0.005, 0.25, 50)
    cosmo.compute()
    cosmo.initialize_output(k, z, len(k))

    b1, b2, bG2, bGamma3, cs0, cs2, cs4, Pshot, b4 = params[4], params[5], 0.1, -0.1, 0., 30., 0., 3000., 10.
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)

    # This line is necesary to prevent memory leaks
    cosmo.struct_cleanup()
    return np.concatenate((pk_g0, pk_g2))

def ln_prior(theta):
    for i in range(len(theta)):
        if (theta[i] < cosmo_prior[i,0]) or (theta[i] > cosmo_prior[i,1]):
            return -np.inf
    return 0.

def ln_lkl(theta, pgg, data_vector, decoder, net, vary_covariance):
    P = get_covariance(decoder, net, theta) if vary_covariance == True else P_fid
    #with io.StringIO() as buf, redirect_stdout(buf):
    #    x = data_vector - model_vector(theta, gparams, pgg)
    x = data_vector - model_vector_CLASS_PT(theta)
    lkl = -0.5 * (x.T @ P @ x)
    assert lkl < 0
    del x
    return lkl

def ln_prob(theta, pgg, data_vector, decoder, net, vary_covariance):
    p = ln_prior(theta)
    if p != -np.inf:
        return p + ln_lkl(theta, pgg, data_vector, decoder, net, vary_covariance)
    else: return p

def Metropolis_Hastings(theta, theta_std, N, NDIM, pgg, data_vector, decoder, net, vary_covariance, resume, save_str):
    """
    runs an mcmc based on metropolis hastings
    """
    if resume == False:
        acceptance_rate = np.zeros(N)
        num_accept = 0
        chain = np.zeros((N, NDIM))
        start = 0
    else:
        file = np.load("Data/"+save_str)
        chain = file["chain"]
        acceptance_rate = file["rate"]
        start = np.where(acceptance_rate[(acceptance_rate != 0)])[0][-1] + 1
        theta = chain[start]
        num_accept = acceptance_rate[start] * (start+1)
        del file

    prob_old = ln_prob(theta, pgg, data_vector, decoder, net, vary_covariance)
    for i in tqdm(range(start, N)):
        # STEP 1: save current state to the chain
        chain[i] = theta

        # STEP 2: generate a new potential move
        theta_new = theta + (theta_std * np.random.normal(size=(NDIM)))

        # STEP 3: determine if we move to the new position based on ln_prob
        prob_new = ln_prob(theta_new, pgg, data_vector, decoder, net, vary_covariance)
        p = np.random.uniform()

        #print(prob_new, np.exp(prob_new))
        if p < min(np.exp(prob_new - prob_old), 1):
            theta = theta_new
            num_accept += 1

        acceptance_rate[i] = num_accept / (i+1.)
        prob_old = prob_new

        if i % 1000 == 0:
            np.savez("Data/"+save_str, chain=chain, rate=acceptance_rate)

    return chain, acceptance_rate

def main():

    if len(sys.argv) != 3:
        print("USAGE: python Likelihood.py <vary covariance> <resume>")
        return
    vary_covariance = True if sys.argv[1].lower() == "true" else False
    resume = True if sys.argv[2].lower() == "true" else False

    print("Running MCMC with varying covariance: " + str(vary_covariance))
    if resume == True:
        print("Resuming previous mcmc run...")

    N    = 60000
    NDIM = 6

    # Make sure that the covariance matrix we're using is positive definite
    if vary_covariance == False:
        try:
            L = np.linalg.cholesky(P_fid)
        except np.linalg.LinAlgError as err:
            print("ERROR! Fiducial percision matrix is not positive definite! Exiting...")
            return
        eigen, _ = np.linalg.eig(P_fid)
        if np.all(eigen >= 0.) == False:
            print("ERROR! Fiducial percision matrix is not positive definite! Exiting...")
            return
        if np.all(P_fid != np.nan) == False:
            print("ERROR: NaN values in the percision matrix! Exiting...")
            return

    #pgg = pkmu_hod()
    pgg = None

    # Setup data vector as the Pk at the fiducial cosmology
    data_vector = model_vector_CLASS_PT(cosmo_fid)
    #data_vector = model_vector(cosmo_fid, gparams, pgg)

    #Cov_emu = PCA_emulator()
    decoder, net = CovNet_emulator()

    if vary_covariance == True:
        theta_std = np.array([1., 0.003, 0.0003, 0.25, 0.03, 0.02])
        save_str = "mcmc_chains_VAE.npz"
    else: 
        theta_std  = np.array([1., 0.003, 0.0003, 0.25, 0.1, 0.25])
        if use_T0 == True:
            save_str = "mcmc_chains_T0.npz"
        else:
            save_str = "mcmc_chains_no_t0.npz"

    # Starting position of the emcee chain
    theta0 = cosmo_fid + (theta_std * np.random.normal(size=(NDIM)))

    # tracemalloc.start()
    # z = 0.58
    # snapshot1 = tracemalloc.take_snapshot()
    # for i in range(50):
    #     k = np.linspace(0.005, 0.25, 50)
    #     cosmo.set({'output':'mPk',
    #             'non linear':'PT',
    #             'IR resummation':'Yes',
    #             'Bias tracers':'Yes',
    #             'cb':'Yes', # use CDM+baryon spectra
    #             'RSD':'Yes',
    #             'AP':'Yes', # Alcock-Paczynski effect
    #             'Omfid':'0.31', # fiducial Omega_m
    #             'PNG':'No', # single-field inflation PNG
    #             'FFTLog mode':'FAST',
    #             'A_s':np.exp(cosmo_fid[3])/1e10,
    #             'n_s':0.9649,
    #             'tau_reio':0.052,
    #             'omega_b':cosmo_fid[2],
    #             'omega_cdm':cosmo_fid[1],
    #             'h':cosmo_fid[0] / 100.,
    #             'YHe':0.2425,
    #             'N_ur':2.0328,
    #             'N_ncdm':1,
    #             'm_ncdm':0.06,
    #             'z_pk':z
    #             }) 
    #     cosmo.compute()
    #     cosmo.initialize_output(k, z, len(k))
    #     b1, b2, bG2, bGamma3, cs0, cs2, cs4, Pshot, b4 = cosmo_fid[4], cosmo_fid[5], 0.1, -0.1, 0., 30., 0., 3000., 10.
    #     pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    #     pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)
    #     #cosmo.empty()
    #     cosmo.struct_cleanup()
    # snapshot2 = tracemalloc.take_snapshot()

    # top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    # print("[ Top 10 differences ]")
    # for stat in top_stats[:10]:
    #     print(stat)
    chain, acceptance_rate = Metropolis_Hastings(theta0, theta_std, N, NDIM, pgg, data_vector, decoder, net, vary_covariance, resume, save_str)
    np.savez("Data/"+save_str, chain=chain, rate=acceptance_rate)

if __name__ == '__main__':
    main()