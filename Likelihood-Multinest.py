import numpy as np
from numpy import pi, cos
import pymultinest
import threading, subprocess
from classy import Class
import torch
import sys, os, io, time, math
from scipy.stats import norm
sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
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

cosmo_prior = [[52, 90],     #H0
               [0.002, 0.3],  #omch2
               [0.005, 0.08], #ombh2
               [0.3, 1.6],    #A / A_planck
               #[0.9, 1.1],    #ns
               [1, 4],        #b1
               norm(0, 1),        #b2 (gaussian)
               norm(0, 1),        #bGamma2 (gaussian)
               norm(0, 30),       #c0 (gaussian)
               norm(0, 30),       #c2 (gaussian)
               norm(500, 500),    #cbar (gaussian)
               norm(0, 5000)      #Pshot (gaussian)
               ]

A_planck = 3.0448
ns_planck = 0.9649
omch2_planck = 0.02237

# fiducial taken to be the cosmology used to generate Patchy mocks
#                     H0,   omch2,  ombh2,  A,    ns,     b1,     b2      bG2, c0, c2,  cbar, Pshot
#cosmo_fid = np.array([67.77,0.11827,0.02214,1.016,0.9611, 1.9485,-0.5387, 0.1, 5., 15., 100, 5e3])
cosmo_fid = np.array([67.77,0.11827,0.02214,1.016, 1.9485,-0.5387, 0.1, 5., 15., 100, 5e3])

redshift = 0.61

cosmo = Class()

W = pk_tools.read_matrix(BOSS_dir+"W_CMASS_North.matrix")
M = pk_tools.read_matrix(BOSS_dir+"M_CMASS_North.matrix")

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

def model_vector_CLASS_PT(params, cosmo):
    z = 0.61

    As = params[3] * A_planck
    cosmo.set({'output':'mPk',
            'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes', # use CDM+baryon spectra
            'RSD':'Yes',
            'AP':'Yes', # Alcock-Paczynski effect
            'Omfid':'0.307115', # fiducial Omega_m
            'PNG':'No', # single-field inflation PNG
            'FFTLog mode':'FAST',
            'A_s':np.exp(As)/1e10,
            'n_s':ns_planck,
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
    #k = np.linspace(0.005, 0.245, 25)
    try:    cosmo.compute()
    except: return [np.nan]

    cosmo.initialize_output(k, z, len(k))

    b1 = params[4] / np.sqrt(params[3])
    b2 = params[5] / np.sqrt(params[3])
    bG2 = params[6] / np.sqrt(params[3])
    bGamma3 = 0
    cs4 = -5
    # NOTE: I'm pretty sure b4 is actually cbar
    #b4 = 100. # from CLASS-PT Notebook (I don't think Wadekar varies this)
    cs0, cs2, b4, Pshot = params[7], params[8], params[9], params[10]
    #b2, bG2, cs0, cs2, Pshot = -0.5387, 0.1, 5., 15., 5e3
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)
    pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, b4)

    # Convolve with the window function
    model_vector = np.concatenate((pk_g0, pk_g2, pk_g4))
    model_vector = np.matmul(M, model_vector)
    model_vector = np.matmul(W, model_vector)

    # This line is necesary to prevent memory leaks
    cosmo.struct_cleanup()

    #return np.concatenate([pk_g0, pk_g2])
    return np.concatenate([model_vector[0:25], model_vector[80:105]])

#pk_dict = pk_tools.read_power(BOSS_dir+"P_CMASS_North.dat" , combine_bins =10)
#data_vector = np.concatenate([pk_dict["pk0"], pk_dict["pk2"]])
data_vector = model_vector_CLASS_PT(cosmo_fid, cosmo)

def get_covariance(decoder, net, theta):
    # first convert theta to the format expected by our emulators
    params = torch.tensor([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]]).float()
    features = net(params); C = decoder(features.view(1,10)).view(50, 50)
    C = CovNet.corr_to_cov(C).cpu().detach().numpy()
    return C

def prior(cube, ndim, nparams):
    """
    Get prior bounds
    """
    for i in range(ndim):
        if i < 5:
            cube[i] = cube[i] * (cosmo_prior[i][1] - cosmo_prior[i][0]) + cosmo_prior[i][0]
        else:
            cube[i] = cosmo_prior[i].ppf(cube[i])

    # calculate Omega_m (TODO: Also calculate sigma8)
    #params = np.append(params, (params[1] + params[2] + 0.00064) / (params[0]/100)**2)
    #return params


def ln_lkl(theta, ndim, nparams):
    #C = get_covariance(decoder, net, cube) if vary_covariance else C_fixed
    C = get_covariance(Cov_emu, theta) if vary_covariance else C_fid
    P = np.linalg.inv(C)
    x = data_vector - model_vector_CLASS_PT(theta, cosmo)
    lkl = -0.5 * np.matmul(x.T, np.matmul(P, x))

    # Current workaround - if model vector is invalid for some reason, return a huge likelihood
    # this "should" be ok, since CLASS-PT only fails for parameters far from the fiducial value
    if True in np.isnan(x):
        return -1e10

    assert lkl < 0
    return lkl

def main():
    # number of dimensions our problem has
    print("Running MCMC with varying covariance: " + str(vary_covariance))
    print("Using T0 term of the covariance: " + str(use_T0))

    parameters = ["H0", "omch2","ombh2", "As", "b1","b2", "bGamma3", "cs0", "cs2", "cbar", "Pshot"]
    n_params = len(parameters)
    # name of the output files
    if vary_covariance == True:
        prefix = "chains/Multinest/test-"
    else: 
        if use_T0 == True: prefix = "chains/Multinest/test-"
        else: prefix = "chains/Multinest/test-"

    # https://arxiv.org/pdf/0809.3437.pdf
    t1 = time.time()
    #progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename=prefix); progress.start()
    #threading.Timer(2, show, [prefix+"phys_live.points.pdf"]).start() # delayed opening
    # run MultiNest
    pymultinest.run(ln_lkl, prior, n_params, outputfiles_basename=prefix, log_zero=-1e9, write_output=True,
                    sampling_efficiency = 1, n_live_points=400, resume=resume, verbose=True)
    #progress.stop()
    t2 = time.time()
    print("Done!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()
