import emcee
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, math
import torch
from tqdm import tqdm
from torch.multiprocessing import Pool, set_start_method
from getdist import plots, MCSamples
import getdist

sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet

vary_covariance = True

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set/"
data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
BOSS_dir = "/home/joeadamo/Research/Data/BOSS-DR12/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"

try:
   set_start_method('spawn')
except:
   pass

C_fixed = np.loadtxt("/home/joeadamo/Research/Data/CovA-survey.txt")

cosmo_prior = np.array([[66.5, 75.5],
                        [0.10782, 0.13178],
                        [0.0211375, 0.0233625],
                        [1.1885e-9, 2.031e-9],#[2.4752, 3.7128],
                        [1.806, 2.04],
                        [-2.962, 0.458]])

cosmo_fid = np.array([70,0.1198,0.02225,2e-9,2.0,0.])

gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}
redshift = 0.5

def PCA_emulator():
    """
    Sets up the PCA covariance matrix emulator to use in likelihood analysis
    """
    N_C = 100
    C_PCA = np.zeros((N_C, 100, 100))
    params_PCA = np.zeros((N_C, 6))
    for i in range(N_C):
        temp = np.load(PCA_dir+"CovA-"+f'{i:04d}'+".npz")
        params_PCA[i] = np.delete(temp["params"], 4)
        C_PCA[i] = torch.from_numpy(temp["C"])
    
        # if the matrix doesn't match the transpose close enough, manually flip over the diagonal
        try:
            np.testing.assert_allclose(C_PCA[i], C_PCA[i].T, err_msg="covariance must match transpose")
        except AssertionError:
            L = np.tril(C_PCA[i])
            U = np.tril(C_PCA[i], k=-1).T
            C_PCA[i] = L + U
    Emu = ce.CovEmu(params_PCA, C_PCA, NPC_D=20, NPC_L=20)
    return Emu

def get_covariance(Emu, theta):
    # first convert theta to the format expected by our emulators
    params = np.array([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]])
    C = Emu.predict(params)
    return C

def model_vector(params, gparams, pgg):
    """
    Calculates the model vector using Yosuke's galaxy power spectrum emulator
    """
    #print(params)
    h = params[0] / 100
    omch2 = params[1]
    ombh2 = params[2]
    #assert omch2 <= 0.131780
    As = np.log(1e10 * params[3])
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

    pgg.set_cosmology(cparams, redshift)
    pgg.set_galaxy(gparams)
    P0_emu = pgg.get_pl_gg_ref(0, k, alpha_perp, alpha_para, name='total')
    P2_emu = pgg.get_pl_gg_ref(2, k, alpha_perp, alpha_para, name='total')
    return np.concatenate((P0_emu, P2_emu))

def ln_prior(theta):
    for i in range(len(theta)):
        if (theta[i] < cosmo_prior[i,0]) or (theta[i] > cosmo_prior[i,1]):
            return -np.inf
    return 0.

def ln_lkl(theta, pgg, data_vector, emu, vary_covariance):
    C = get_covariance(emu, theta) if vary_covariance else C_fixed
    P = np.linalg.inv(C)
    x = model_vector(theta, gparams, pgg) - data_vector
    lkl = -0.5 * np.matmul(x.T, np.matmul(P, x))
    return lkl

def ln_prob(theta, pgg, data_vector, emu, vary_covariance):
    p = ln_prior(theta)
    if p != -np.inf:
        return p * ln_lkl(theta, pgg, data_vector, emu, vary_covariance)
    else: return p

def main():

    print("Running MCMC with varying covariance: " + str(vary_covariance))
    N_MCMC        = 4000
    N_WALKERS     = 40
    NDIM_SAMPLING = 6

    P_BOSS = np.loadtxt(BOSS_dir+"Cl-BOSS-DR12.dat")
    pgg = pkmu_hod()

    Cov_emu = PCA_emulator()

    data_vector = np.concatenate((P_BOSS[1], P_BOSS[2]))
    theta0    = cosmo_fid
    theta_std = np.array([1., 0.001, 0.0001, 0.01 * 2e-9, 0.01, 0.01])

    # Starting position of the emcee chain
    pos0 = theta0[np.newaxis] + theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, NDIM_SAMPLING))
    
    t1 = time.time()
    #with Pool() as pool:
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, ln_prob, args=(pgg, data_vector, Cov_emu, vary_covariance))#, pool=pool)
    sampler.run_mcmc(pos0, N_MCMC, progress=True)
    t2 = time.time()
    print("Done!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

    np.save('Data/mcmc_chains.npz', sampler.chain)

if __name__ == '__main__':
    main()