# This script is basically CovaPT's jupyter notebook, but in script form so you can more easily run it

import time, os, sys, warnings, math
import numpy as np
from scipy.stats import qmc

from multiprocessing import Pool
from itertools import repeat
from mpi4py import MPI

sys.path.append('/home/u12/jadamo/')
#sys.path.append("/home/joeadamo/Research")
import CovaPT

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------
As_planck = 3.0447
ns_planck = 0.9649
ombh2_planck = 0.02237

# wether or not to also vary nuisance parameters
# if you want to emulate both gaussian and non-gaussian parts of the covariance matrix, set this to true
# NOTE: This doubles the dimensionality of the parameter space, so you should also increase the 
# number of samples to compensate
vary_nuisance = False
# wether or not to sample from an external set of parameters instead
# if this is true you should also specify the total number of params below
load_external_params = True

#N = 150400
N = 10000
#N = 16
N_PROC = 94
#N_PROC=4

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

dire='/home/u12/jadamo/CovaPT/Example-Data/'
#home_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
home_dir = "/home/u12/jadamo/CovNet/Inportance-Set/"
#dire='/home/joeadamo/Research/CovaPT/Example-Data/'
#home_dir = "/home/joeadamo/Research/CovNet/Data/Inportance-Set/"

def Latin_Hypercube(N, vary_nuisance=False, vary_ombh2=False, vary_ns=False):
    """
    Generates a random latin hypercube sample of the parameters for generating the training set
    @param N {int} the number of samples to generate
    """
    # TODO: impliment varying ombh2 and ns

    n_dim = 6
    if vary_nuisance == True: n_dim += 5
    #if vary_ombh2 == True:    n_dim += 1
    #if vary_ns == True:       n_dim += 1

    # bounds either taken from Wadekar et al (2020) or assumed to be "very wide"
    # NOTE: H0, A, and ombh2 have no assumed priors in that paper, so I chose an arbitrary large range
    # that minimizes the amount of failures when computing power spectra
    # NOTE: ns and omega_b have assumed values, as they claim using Planck priors makes no difference.
    # I'll therefore try to chose a range based on those priors found from https://wiki.cosmos.esa.int/planckpla/index.php/Cosmological_Parameters 
    # For As, the reference value is taken from https://arxiv.org/pdf/1807.06209.pdf table 1 (the best fit column), 
    # since Wadekar uses A = As / As_planck
    # ---Cosmology parameters sample bounds---
    # omch2_bounds = [0.004, 0.3]   # Omega_cdm h^2
    # A_bounds     = [0.2, 1.75]    # Ratio of Amplitude of Primordial Power spectrum (As / As_planck)
    H0_bounds    = [50, 100]      # Hubble constant
    omch2_bounds = [0.01, 0.3]    # Omega_cdm h^2
    A_bounds     = [0.25, 1.65]   # Ratio of Amplitude of Primordial Power spectrum (As / As_planck)
    b1_bounds    = [1, 4]         # Linear bias
    b2_bounds    = [-5, 3]        # Quadratic bias?
    bG2_bounds   = [-4, 4]        # 

    ombh2_bounds = [0.005, 0.08]  # Omega b h^2
    ns_bounds    = [0.9, 1.1]     # Spectral index
    # nuisance parameter sample bounds, should you chose to vary these when generating your training set
    cs0_bounds   = [-120, 120]
    cs2_bounds   = [-120, 120]
    cbar_bounds  = [-1500, 2500]
    Pshot_bounds = [-20000, 20000]

    # sample the distribution of points using a Latin Hypercube
    # specify a seed so that each rank generates the SAME latin hypercube
    sampler = qmc.LatinHypercube(d=n_dim, seed=81341234)
    dist = sampler.random(n=N)

    # ---Cosmology parameters---
    H0 = dist[:,0]*(H0_bounds[1] - H0_bounds[0]) + H0_bounds[0]
    omch2 = dist[:,1]*(omch2_bounds[1] - omch2_bounds[0]) + omch2_bounds[0]
    A = dist[:,2]*(A_bounds[1] - A_bounds[0]) + A_bounds[0]
    b1 = dist[:,3]*(b1_bounds[1] - b1_bounds[0]) + b1_bounds[0]
    b2 = dist[:,4]*(b2_bounds[1] - b2_bounds[0]) + b2_bounds[0]
    bG2 = dist[:,5]*(bG2_bounds[1] - bG2_bounds[0]) + bG2_bounds[0]

    # if vary_ombh2:
    #     ombh2 = dist[:,2]*(ombh2_bounds[1] - ombh2_bounds[0]) + ombh2_bounds[0]
    # if vary_ns:
    #     ns = dist[:,4]*(ns_bounds[1] - ns_bounds[0]) + ns_bounds[0]
    if vary_nuisance == True:
        cs0 = dist[:,6]*(cs0_bounds[1] - cs0_bounds[0]) + cs0_bounds[0]
        cs2 = dist[:,7]*(cs2_bounds[1] - cs2_bounds[0]) + cs2_bounds[0]
        cbar = dist[:,8]*(cbar_bounds[1] - cbar_bounds[0]) + cbar_bounds[0]
        Pshot = dist[:,9]*(Pshot_bounds[1] - Pshot_bounds[0]) + Pshot_bounds[0]

        samples = np.vstack((H0, omch2, A, b1, b2, bG2, cs0, cs2, cbar, Pshot)).T
        header_str = "H0, omch2, A, b1, b2, bG2, cs0, cs2, cbar, Pshot"
    else:
        header_str = "H0, omch2, A, b1, b2, bG2"
        samples = np.vstack((H0, omch2, A, b1, b2, bG2)).T

    np.savetxt("Sample-params.txt", samples, header=header_str)
    #return samples

def load_samples(N):
    """
    Loads in a set of parameters to generate covariance matrices from
    """
    samples = np.loadtxt(home_dir+"importance_params.txt", skiprows=1)
    samples = samples[:N, :]
    return samples

def CovAnalytic(H0, omch2, A, b1, b2, bG2, cs0, cs2, cbar, Pshot, z, i):
    """
    Generates and saves the Non-Gaussian term of the analytic covariance matrix. This function is meant to be run
    in parallel.
    """

    params = np.array([H0, omch2, A, b1, b2, bG2, cs0, cs2, cbar, Pshot])

    Mat_Calc = CovaPT.Analytic_Covmat(z)

    # calculate the covariance matrix
    C_G, Pk_galaxy = Mat_Calc.get_gaussian_covariance(params, return_Pk=True)
    if True in np.isnan(C_G):
        print("idx", i, "failed to compute power spectrum! skipping...")
        return -1

    C_SSC, C_T0 = Mat_Calc.get_non_gaussian_covariance(params)

    # Test that the matrix we calculated is positive definite. It it isn't, then skip
    try:
        L = np.linalg.cholesky(C_G + C_SSC + C_T0)

        # save results to a file for training
        idx = f'{i:05d}'
        params_save = np.array([H0, omch2, A, b1, b2, bG2])
        np.savez(home_dir+"CovA-"+idx+".npz", params=params_save, Pk=Pk_galaxy, C_G=C_G, C_SSC=C_SSC, C_T0 = C_T0)
        return 0
    except:
        print("idx", i, "is not positive definite! skipping...")
        return -2

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Varying nuisance parameters:", vary_nuisance)
        print("Loading external set of parameters:", load_external_params)

    # TEMP: ignore integration warnings to make output more clean
    warnings.filterwarnings("ignore")

    t1 = time.time(); t2 = t1

    if rank == 0:
        # generate samples and sve them (only one rank does this!)
        Latin_Hypercube(N, vary_nuisance)
    comm.Barrier()
    # generate samples (done on each rank for simplicity)
    if load_external_params == False: file = "Sample-params.txt"
    else:                             file = home_dir+"inportance_params.txt"

    # Split up samples to multiple MPI ranks
    # Aparently MPI scatter doesn't work on Puma, so this uses a different way
    assert N % size == 0
    offset = int((N / size) * rank) + 1
    data_len = int(N / size)
    sample = np.loadtxt(file, skiprows=offset, max_rows=data_len)

    # ---Cosmology parameters---
    H0 = sample[:,0]
    omch2 = sample[:,1]
    A = sample[:,2]
    b1 = sample[:,3]
    b2 = sample[:,4]
    bG2 = sample[:,5]

    # sample nuisance parameters, or set them to a constant
    if vary_nuisance == True:
        cs0 = sample[:6]
        cs2 = sample[:7]
        cbar = sample[:8]
        Pshot = sample[:9]
    else:
        cs0 = np.zeros(data_len)
        cs2 = np.zeros(data_len)
        cbar = np.ones(data_len)*500.
        Pshot = np.zeros(data_len)

    z = 0.61
    # split up workload to different nodes
    i = np.arange(offset, offset+data_len, dtype=np.int)

    # initialize pool for multiprocessing
    print("Rank", rank, "beginning matrix calculations...")
    t1 = time.time()
    fail_compute_sub = 0
    fail_posdef_sub = 0
    with Pool(processes=N_PROC) as pool:
        for result in pool.starmap(CovAnalytic, zip(H0, omch2, A, b1, b2, bG2, cs0, cs2, cbar, Pshot, repeat(z), i)):
            if result == -1: fail_compute_sub+=1
            if result == -2: fail_posdef_sub+=1

    t2 = time.time()
    print("Rank {:0.0f} is Done! Took {:0.0f} hours {:0.0f} minutes".format(rank, math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("{:0.0f} matrices failed to compute power spectra".format(fail_compute_sub))
    print("{:0.0f} matrices were not positive definite".format(fail_posdef_sub))
    comm.Barrier()

    # gather reasons for failure
    fail_compute = comm.reduce(fail_compute_sub, op=MPI.SUM, root=0)
    fail_posdef = comm.reduce(fail_posdef_sub, op=MPI.SUM, root=0)

    if rank == 0:
        t2 = time.time()
        print("\n All ranks done! Took {:0.0f} hours {:0.0f} minutes".format(math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))

        # there (should be) no directories in the output directory, so no need to loop over the files
        files = os.listdir(home_dir)
        num_success = len(files)
        print("Succesfully made {:0.0f} / {:0.0f} matrices ({:0.2f}%)".format(num_success, N, 100.*num_success/N))
        print("{:0.0f} matrices failed to compute power spectra".format(fail_compute))
        print("{:0.0f} matrices were not positive definite".format(fail_posdef))

if __name__ == "__main__":
    main()
