import numpy as np
from numpy import pi, cos
import pymultinest
import threading, subprocess
from classy import Class
import torch
import sys, os, io, time, math
from contextlib import redirect_stdout
sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet
sys.path.append('/home/joeadamo/Research/Software')
from pk_tools import pk_tools

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-HighZ-NGC/"
data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
BOSS_dir = "/home/joeadamo/Research/Data/BOSS-DR12/Updated/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"

# Flags to be changed by the user
use_T0 = True
vary_covariance = False
resume = False

C_fid_file = np.load(data_dir+"Cov_Fid.npz")
if use_T0 == True:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"] + C_fid_file["C_T0"]
else:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"]

P_fid = np.linalg.inv(C_fid)

cosmo_prior = np.array([[60, 75],
                        [0.09, 0.15],
                        [0.02, 0.025],
                        [2.4, 3.8],
                        [0.94, 0.99],
                        [1.7, 2.45],
                        [-3.5, 0.75]])

#                     H0,omch2, ombh2,  As,  ns, b1,     b2
cosmo_fid = np.array([67.8,0.1190,0.02215,3.094,0.9649, 1.9485,-0.5387])

gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}
redshift = 0.61

W = pk_tools.read_matrix(BOSS_dir+"W_CMASS_North.matrix")
M = pk_tools.read_matrix(BOSS_dir+"M_CMASS_North.matrix")

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
    Emu = ce.CovEmu(params_PCA, C_PCA, NPC_D=20, NPC_L=20)
    return Emu

def CovNet_emulator():
    """
    Returns a neural net covariance matrix emulator
    """
    VAE = CovNet.Network_VAE().to(CovNet.try_gpu());       VAE.eval()
    decoder = CovNet.Block_Decoder().to(CovNet.try_gpu()); decoder.eval()
    net = CovNet.Network_Features(6, 10).to(CovNet.try_gpu())
    VAE.load_state_dict(torch.load(data_dir+'Correlation-decomp/network-VAE.params', map_location=CovNet.try_gpu()))
    decoder.load_state_dict(VAE.Decoder.state_dict())
    net.load_state_dict(torch.load(data_dir+"Correlation-decomp/network-features.params", map_location=CovNet.try_gpu()))

    return decoder, net

# Define these as global variables so that we don't need to pass them into ln_lkl
#decoder, net = CovNet_emulator()
#Cov_emu = PCA_emulator()
Cov_emu = None

def model_vector(params, gparams, pgg):
    """
    Calculates the model vector using Yosuke's galaxy power spectrum emulator
    """
    #print(params)
    h = params[0] / 100
    omch2 = params[1]
    ombh2 = params[2]
    #assert omch2 <= 0.131780
    #As = np.log(1e10 * params[3])
    As = params[3]
    #assert As >= 2.47520
    ns = 0.965
    Om0 = (omch2 + ombh2 + 0.00064) / (h**2)
    
    # rebuild parameters into correct format (ombh2, omch2, 1-Om0, ln As, ns, w)
    cparams = np.array([ombh2, omch2, 1-Om0, As, ns, -1])
    redshift = 0.58
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
    z = 0.61
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
            'n_s':params[4],
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
    k = np.linspace(0.005, 0.395, 400)
    cosmo.compute()
    cosmo.initialize_output(k, z, len(k))

    b1, b2, bG2, bGamma3, cs0, cs2, cs4, Pshot, b4 = params[5], params[6], 0.1, -0.1, 5., 15., -5., 1.3e3, 100.
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)
    pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, b4)

    # Convolve with the window function
    model_vector = np.concatenate((pk_g0, pk_g2, pk_g4))
    model_vector = np.matmul(M, model_vector)
    model_vector = np.matmul(W, model_vector)

    # This line is necesary to prevent memory leaks
    cosmo.struct_cleanup()

    return np.concatenate([model_vector[0:40], model_vector[80:120]])

pk_dict = pk_tools.read_power(BOSS_dir+"P_CMASS_North.dat" , combine_bins =10)
data_vector = np.concatenate([pk_dict["pk0"], pk_dict["pk2"]])

# def get_covariance(decoder, net, theta):
#     # first convert theta to the format expected by our emulators
#     params = torch.tensor([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]]).float()
#     features = net(params); C = decoder(features.view(1,10)).view(100, 100)
#     C = CovNet.corr_to_cov(C).cpu().detach().numpy()
#     return C

def get_covariance(Emu, theta):
    # first convert theta to the format expected by our emulators
    params = np.array([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]])
    C = Emu.predict(params)
    return C

def prior(cube, ndim, nparams):
    for i in range(6):
        cube[i] = cube[i] * (cosmo_prior[i, 1] - cosmo_prior[i, 0]) + cosmo_prior[i, 0]

def ln_lkl(cube, ndim, nparams):
    #C = get_covariance(decoder, net, cube) if vary_covariance else C_fixed
    C = get_covariance(Cov_emu, cube) if vary_covariance else C_fid
    P = np.linalg.inv(C)
    x = data_vector - model_vector_CLASS_PT(cube)
    lkl = -0.5 * np.matmul(x.T, np.matmul(P, x))
    assert lkl < 0
    return lkl

def main():
    # number of dimensions our problem has
    print("Running MCMC with varying covariance: " + str(vary_covariance))
    print("Using T0 term of the covariance: " + str(use_T0))

    parameters = ["H0", "omch2","ombh2", "As", "ns", "b1","b2"]
    n_params = len(parameters)
    # name of the output files
    if vary_covariance == True:
        prefix = "chains/Varied-"
    else: 
        if use_T0 == True: prefix = "chains/Fixed-"
        else: prefix = "chains/Fixed-"

    # https://arxiv.org/pdf/0809.3437.pdf
    t1 = time.time()
    #progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename=prefix); progress.start()
    #threading.Timer(2, show, [prefix+"phys_live.points.pdf"]).start() # delayed opening
    # run MultiNest
    pymultinest.run(ln_lkl, prior, n_params, outputfiles_basename=prefix, 
                    sampling_efficiency = 1, n_live_points=400, resume=resume, verbose=True)
    #progress.stop()
    t2 = time.time()
    print("Done!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()
