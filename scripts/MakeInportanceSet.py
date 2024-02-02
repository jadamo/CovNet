# This script is basically CovaPT's jupyter notebook, but in script form so you can more easily run it

import time, os, sys, warnings, math
import numpy as np
from scipy.stats import qmc

from multiprocessing import Pool
from itertools import repeat
from mpi4py import MPI

sys.path.append('/home/u12/jadamo/')
#sys.path.append("/home/joeadamo/Research")
import src.CovaPT as CovaPT

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------
As_planck = 3.0447
ns_planck = 0.9649
ombh2_planck = 0.02237

#N = 150400
N = 72400
#N = 8
#N = 4
#N = 16
N_PROC = 94
#N_PROC=4

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

dire='/home/u12/jadamo/CovaPT/Example-Data/'
#home_dir = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
#home_dir = "/xdisk/timeifler/jadamo/Training-Set-HighZ-NGC/"
home_dir = "/xdisk/timeifler/jadamo/Inportance-Set-1/"
BOSS_dir = "/home/u12/jadamo/software/montepython/data/BOSS_DR12/Updated/"
#dire='/home/joeadamo/Research/CovaPT/Example-Data/'
#home_dir = "/home/joeadamo/Research/CovNet/Data/Inportance-Set/"

def CovAnalytic(H0, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot, z, i):
    """
    Generates and saves the Non-Gaussian term of the analytic covariance matrix. This function is meant to be run
    in parallel.
    """

    params = np.array([H0, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot])
    Mat_Calc = CovaPT.Analytic_Covmat(z, window_dir="/home/u12/jadamo/CovaPT/Example-Data/")

    # calculate the covariance matrix
    C_G = Mat_Calc.get_gaussian_covariance(params, return_Pk=False)
    if np.any(np.isnan(C_G)) == True:
        print("idx", i, "failed to compute power spectrum! skipping...")
        return -1

    C_SSC, C_T0 = Mat_Calc.get_non_gaussian_covariance(params)

    # Test that the matrix we calculated is positive definite. It it isn't, then skip
    try:
        L = np.linalg.cholesky(C_G + C_SSC + C_T0)
    except:
        return -2

    try:
        # save results to a file for trainin
        C_marg, model_vector, om0, s8 = Mat_Calc.get_marginalized_covariance(params, C_G+C_SSC+C_T0, BOSS_dir)
        idx = f'{i:05d}'
        params_save = np.array([H0, omch2, As, b1, b2, bG2, om0, s8])
        np.savez(home_dir+"CovA-"+idx+".npz",
                params=params_save, model_vector=model_vector, C_G=C_G, C_NG=C_SSC+C_T0, C_marg=C_marg)
        return 0
    except:
        print("idx", i, "failed to marginalize covariance! saving what we have")
        idx = f'{i:05d}'
        params_save = np.array([H0, omch2, As, b1, b2, bG2])
        np.savez(home_dir+"CovA-"+idx+".npz",
                    params=params_save, C_G=C_G, C_NG=C_SSC + C_T0)
        return -3

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # TEMP: ignore integration warnings to make output more clean
    warnings.filterwarnings("ignore")

    t1 = time.time(); t2 = t1

    # generate samples (done on each rank for simplicity)
    file = home_dir+"inportance-params-T1.txt"
    assert os.path.exists(file)
    
    if rank == 0:
        print("Saving matrices to", home_dir)

    # Split up samples to multiple MPI ranks
    # but only read into one rank to prevent wierd race conditions
    # if rank == 0:
    #     sample_full = np.loadtxt(file, skiprows=1)[:N]
    # else: sample_full = np.zeros((N, 6))
    # print("Rank", rank, "about to hit the barrier!")
    # comm.Barrier()
    # print("Rank", rank, "passed the barrier!")
    # comm.Bcast(sample_full, root=0)
    # if rank == 0:
    #     print("parameters succesfully broadcasted!")

    assert N % size == 0
    offset = int((N / size) * rank)
    data_len = int(N / size)
    #sample = sample_full[offset:offset+data_len,:]
    sample = np.loadtxt(file, skiprows=offset+1, max_rows=data_len)
    assert sample.shape[0] <= data_len
    comm.Barrier()
    if rank == 5:
        sample_test = np.loadtxt(file, skiprows=offset+1, max_rows=data_len)
        np.testing.assert_equal(sample, sample_test)
        print("parameters succesfully read in!")

    # ---Cosmology parameters---
    H0 = sample[:,0]
    omch2 = sample[:,1]
    As = sample[:,2]
    b1 = sample[:,3]
    b2 = sample[:,4]
    bG2 = sample[:,5]

    # set nuisance parameters to a constant
    cs0 = np.zeros(data_len)
    cs2 = np.zeros(data_len)
    cbar = np.ones(data_len)*500.
    Pshot = np.zeros(data_len)

    z = 0.61
    # split up workload to different nodes
    i = np.arange(offset, offset+data_len, dtype=np.int32)

    # initialize pool for multiprocessing
    print("Rank", rank, "beginning matrix calculations...")

    t1 = time.time()
    fail_compute_sub = 0
    fail_posdef_sub = 0
    fail_marg_sub = 0
    with Pool(processes=N_PROC) as pool:
        for result in pool.starmap(CovAnalytic, zip(H0, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot, repeat(z), i)):
            if result == -1: fail_compute_sub+=1
            if result == -2: fail_posdef_sub+=1
            if result == -3: fail_marg_sub+=1

    t2 = time.time()
    print("Rank {:0.0f} is Done! Took {:0.0f} hours {:0.0f} minutes".format(rank, math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))
    print("{:0.0f} matrices failed to compute power spectra".format(fail_compute_sub))
    print("{:0.0f} matrices were not positive definite".format(fail_posdef_sub))
    comm.Barrier()

    # gather reasons for failure
    fail_compute = comm.reduce(fail_compute_sub, op=MPI.SUM, root=0)
    fail_posdef = comm.reduce(fail_posdef_sub, op=MPI.SUM, root=0)
    fail_marg = comm.reduce(fail_marg_sub, op=MPI.SUM, root=0)

    if rank == 0:
        t2 = time.time()
        print("\n All ranks done! Took {:0.0f} hours {:0.0f} minutes".format(math.floor((t2 - t1)/3600), math.floor((t2 - t1)/60%60)))

        # there (should be) no directories in the output directory, so no need to loop over the files
        files = os.listdir(home_dir)
        num_success = len(files) - 1
        print("Succesfully made {:0.0f} / {:0.0f} matrices ({:0.2f}%)".format(num_success, N, 100.*num_success/N))
        print("{:0.0f} matrices failed to compute power spectra".format(fail_compute))
        print("{:0.0f} matrices were not positive definite".format(fail_posdef))
        print("{:0.0f} matrices couldn't be marginalized".format(fail_marg))

if __name__ == "__main__":
    main()
