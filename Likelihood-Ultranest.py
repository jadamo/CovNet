import numpy as np
import ultranest
import ultranest.stepsampler
from classy import Class
import torch
import sys, os, io, time, math
os.environ["OPENBLAS_NUM_THREADS"] = "4"
from scipy.stats import norm
#sys.path.append('/home/u12/jadamo/')
sys.path.append('/home/joeadamo/Research')
import CovNet, CovaPT
#sys.path.append('/home/u12/jadamo/software')
sys.path.append('/home/joeadamo/Research/Software')
from pk_tools import pk_tools

# training_dir = "/home/u12/jadamo/CovNet/Data/Training-Set-HighZ-NGC/"
# data_dir =  "/home/u12/jadamo/CovNet/Data/"
# PCA_dir = "/home/u12/jadamo/CovNet/Data/PCA-Set/"
# BOSS_dir = "/home/u12/jadamo/CovNet/Data/BOSS-DR12/Updated/"
# CovaPT_dir = "/home/u12/jadamo/CovaPT/Example-Data/"

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
#P_fid = np.linalg.inv(C_fid)


# C_fid = pk_tools.read_matrix(BOSS_dir+"Updated/Cov_CMASS_North.matrix.gz")
# krange = np.linspace(0.005, 0.395, num=40)
# fit_selection = np.logical_and(0<krange,krange<0.25)
# pole_selection = [True, False, True, False, False]
# fit_selection = np.logical_and(np.tile(fit_selection, 5), np.repeat(pole_selection , 40))
# C_fid = C_fid[np.ix_(fit_selection , fit_selection)]

P_fid = np.linalg.inv(C_fid)

cosmo_prior = [[40., 120.],     #H0
               [0.002, 0.3],  #omch2
               #[0.005, 0.08], #ombh2
               [0.1, 4],    #A / A_planck
               #[0.9, 1.1],    #ns
               [1, 4],        #b1
            #    [-4, 4],
            #    [-4, 4],
            #    [-120, 120],
            #    [-120, 120],
            #    [-1500, 2500],
            #    [-20000, 20000]
                norm(0, 1),        #b2 (gaussian)
                norm(0, 1),        #bGamma2 (gaussian)
                norm(0, 30),       #c0 (gaussian)
                norm(0, 30),       #c2 (gaussian)
                norm(500, 500),    #cbar (gaussian)
                norm(0, 5000)      #Pshot (gaussian)
               ]

A_planck = 3.0447
ns_planck = 0.9649
ombh2_planck = 0.02237

# fiducial taken to be the cosmology used to generate Patchy mocks
#                     H0,   omch2,  ombh2,  A,    ns,     b1,     b2      bG2, c0, c2,  cbar, Pshot
#cosmo_fid = np.array([67.77,0.11827,0.02214,1.016,0.9611, 1.9485,-0.5387, 0.1, 5., 15., 100, 5e3])
cosmo_fid = np.array([67.77,0.11827,1.016, 1.9485,-0.5387, 0.1, 5., 15., 100, 5e3])
#cosmo_fid = np.array([67.77, 0.11827])#, 5e3])#, 100, 5e3])
z = 0.61

cosmo = Class()
common_settings = {'output':'mPk',         # what to output
                   'non linear':'PT',      # {None, Halofit, PT}
                   'IR resummation':'Yes',
                   'Bias tracers':'Yes',
                   'cb':'Yes',             # use CDM+baryon spectra
                   'RSD':'Yes',            # Redshift-space distortions
                   'AP':'Yes',             # Alcock-Paczynski effect
                   'Omfid':'0.31',         # fiducial Omega_m
                   'PNG':'No',             # single-field inflation PNG
                   'FFTLog mode':'FAST',
                   'k_pivot':0.05,
                   'P_k_max_h/Mpc':100.,
                   'tau_reio':0.0543,      # ?
                   'YHe':0.2454,           # Helium fraction?
                   'N_ur':2.0328,          # ?
                   'N_ncdm':1,             # 1 massive neutrino
                   'm_ncdm':0.06,          # mass of neutrino (eV)
                   'T_ncdm':0.71611,       # neutrino temperature?
                   }

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

def calc_sigma8(params):
    H0    = params[0]
    omch2 = params[1]
    ombh2 = ombh2_planck
    As    = params[2] * A_planck
    ns    = ns_planck
    # This line is necesary to prevent memory leaks
    cosmo.set(common_settings)
    cosmo.set({'A_s':np.exp(As)/1e10,
               'n_s':ns,
               'omega_b':ombh2,
               'omega_cdm':omch2,
               'H0':H0,
               'z_pk':z
               })  
    try:    
        cosmo.compute()
        s8 = cosmo.sigma8()
    except:
        s8 = -1
    cosmo.struct_cleanup()
    return s8

def model_vector_CLASS_PT(params):

    # unpack / specify parameters
    H0    = params[0]
    omch2 = params[1]
    ombh2 = ombh2_planck
    As    = params[2] * A_planck
    ns    = ns_planck

    b1    = 1.984 #params[3] / np.sqrt(params[2])
    b2    = -0.5387 #params[4] / np.sqrt(params[2])
    bG2   = 0.6 #params[5] / np.sqrt(params[2])
    bGamma3 = 0 # set to 0 since multipoles can only distinguish bG2 + 0.4*bGamma3
    # we're only using monopole+quadropole, so the specific value for this "shouldn't" matter
    cs4 = -5.
    # NOTE: I'm pretty sure b4 is actually cbar
    cs0   = 5 #params[6]
    cs2   = 15 #params[7]
    cbar  = 100 #params[8]
    Pshot = 5e3 #params[9]

    cosmo.set(common_settings)
    cosmo.set({'A_s':np.exp(As)/1e10,
               'n_s':ns,
               'omega_b':ombh2,
               'omega_cdm':omch2,
               'H0':H0,
               'z_pk':z
               })  
    try:    cosmo.compute()
    except: return np.nan

    k = np.linspace(0.005, 0.395, 400)
    #k = np.linspace(0.005, 0.245, 25)

    cosmo.initialize_output(k*cosmo.h(), z, len(k))

    # calculate galaxy power spectrum multipoles
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, cbar)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, cbar)
    pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, cbar)

    # Convolve with the window function
    model_vector = np.concatenate((pk_g0, pk_g2, pk_g4))
    model_vector = np.matmul(M, model_vector)
    model_vector = np.matmul(W, model_vector)

    cosmo.struct_cleanup()

    #return np.concatenate([pk_g0, pk_g2])
    return np.concatenate([model_vector[0:25], model_vector[80:105]])

pk_dict = pk_tools.read_power(BOSS_dir+"P_CMASS_North.dat" , combine_bins =10)
data_vector = np.concatenate([pk_dict["pk0"][:25], pk_dict["pk2"][:25]])
#data_vector = model_vector_CLASS_PT(cosmo_fid)

def get_covariance(decoder, net, theta):
    # first convert theta to the format expected by our emulators
    params = torch.tensor([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]]).float()
    features = net(params); C = decoder(features.view(1,10)).view(50, 50)
    C = CovNet.corr_to_cov(C).cpu().detach().numpy()
    return C

def prior(cube):
    """
    Get prior bounds
    """
    params = cube.copy()
    for i in range(len(params)):
        if i < 4:
            params[i] = cube[i] * (cosmo_prior[i][1] - cosmo_prior[i][0]) + cosmo_prior[i][0]
        else:
            params[i] = cosmo_prior[i].ppf(cube[i])

    # calculate Omega_m
    params = np.append(params, (params[1] + ombh2_planck + 0.00064) / (params[0]/100)**2)
    
    # calculate sigma8
    # NOTE: Due to ultranest's structure we have to call cosmo.compute() here as well as in ln_lkl
    # if we want to keep track of sigma8. This is inefficient, but it's not the end of the world
    s8 = calc_sigma8(params)
    params = np.append(params, s8)

    return params

def ln_lkl(theta):
    """
    Log likelihood function to pass to Ultranest
    """
    # calculate data - model
    x = data_vector - model_vector_CLASS_PT(theta)

    P = get_covariance(Cov_emu, theta) if vary_covariance else P_fid
    lkl = -0.5*np.matmul(x.T, np.matmul(P, x))
    #b2, bG2, cs0, cs2, cbar, Pshot = theta[4], theta[5], theta[6], theta[7], theta[8], theta[9]
    #lkl += (b2 - 0.)**2./1**2. + (bG2 - 0.)**2/1**2. + (cs0)**2/30**2 + cs2**2/30**2 + (cbar-500.)**2/500**2 + (Pshot - 0.)**2./(5e3)**2.
    #lkl *= -0.5
    
    # Current workaround: if model vector is invalid for some reason, return a huge likelihood
    # this "should" be ok, since CLASS-PT only fails for parameters far from the fiducial value
    if True in np.isnan(x):
        return -1e10 # + (theta - cosmo_fid)

    assert lkl < 0
    return lkl

def main():

    print("Running MCMC with varying covariance: " + str(vary_covariance))
    print("Using T0 term of the covariance: " + str(use_T0))

#     params_bad = np.array([54, 1.72222821e-02, 7.86029642e-02, 1.51755551e+00,
#  9.19244016e-01, 2.44988013e+00])
#     print(model_vector_CLASS_PT(params_bad))

    # Sanity check: the fiducial matrix is positive definite
    try:
        L = np.linalg.cholesky(P_fid)
    except:
        print("ERROR! Fiducial matrix is not positive definite! Aborting...")
        return -1

    #param_names = ["H0","omch2", "A", "b1","b2", "bG2", "cs0", "cs2", "cbar", "Pshot"]
    param_names = ["H0", "omch2", "A"]#, "cbar", "Pshot"]

    t1 = time.time()
    sampler = ultranest.ReactiveNestedSampler(param_names, ln_lkl, prior,
                                              derived_param_names=["Omega_0", "sigma8"],
                                              draw_multiple=True,
                                              log_dir="chains/Ultranest/test/", resume="overwrite")

    # params = np.array([67,0.11827,0.02214,1.02,0.9611, 1.9485,-0.5387, 0.1, 5., 15., 5e3])
    # t = time.time()
    # for i in range(100):
    #     dummy = ln_lkl(params)
    # t2 = time.time()
    # print("Likelihood calculation avg time: {:0.3f} s".format((t2 - t)/100))

    # nsteps = len(param_names)
    # # create step sampler:
    # sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    #      nsteps=nsteps,
    #      generate_direction=ultranest.stepsampler.generate_mixture_random_direction)

    # sample nuisance parameters less frequently
    # sampler.stepsampler = ultranest.stepsampler.SpeedVariableRegionSliceSampler([Ellipsis, slice(4,10)])

    sampler.run(min_num_live_points=300, 
                dlogz=0.5, 
                min_ess=300,
                max_ncalls=400000,
                frac_remain=0.75,
                max_num_improvement_loops=3)

    sampler.print_results()

    #sampler.plot()

    t2 = time.time()
    print("Done!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()
