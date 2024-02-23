# This script calculates the window function used by the Gaussian covariance term in CovaPT
# NOTE: It is highly recommended you run this on multiple cpu cores, or on an hpc cluster
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import time, os
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from CovNet import window
from CovNet.config import CovaPT_data_dir

calc_FFTs = False
calc_SSC_window = True
calc_Gaussian_window = False

# String specifying what data chunk this window function was generated with
key='HighZ_NGC'

# Location of BOSS DR12 random catalogs
data_dir = "/home/joeadamo/Research/Data/BOSS-DR12/"

# how many processers to use for the Gaussian window kernels
num_processes = 14

# number of kmodes to sample
# The default used by Jay Wadekar was 25000, which was run on a cluster
kmodes_sampled = 25000

# K bins to generate the Gaussian window function for
k_centers = np.linspace(0.01, 0.19, 10)

def main():
    
    print("Calculating FFT kernels:", calc_FFTs)
    print("Calculating SSC window functions:", calc_SSC_window)
    print("Calculating Gaussian window functions:", calc_Gaussian_window)

    if calc_FFTs or calc_SSC_window:
        survey_kernels = window.Survey_Window_Kernels(0.7, 0.31, key, data_dir)

    if calc_FFTs:
        print("\nStarting FFT calculations...")
        t1 = time.time()
        export = survey_kernels.calc_gaussian_kernels(48, 3750)
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = CovaPT_data_dir+'FFTWinFun_'+key+'.npy'
        np.save(save_file,export)
        print("FFTs saved to", save_file)

    if calc_SSC_window:
        print("\nStarting FFT calculations...")
        t1 = time.time()
        P_W = survey_kernels.calc_SSC_window_function(300, 7200)
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = CovaPT_data_dir+'WindowPowers_'+key+'.npy'
        np.save(save_file,P_W)
        print("SSC window functions saved to", save_file)

    if calc_Gaussian_window:
        print("\nStarting Gaussian window function generation with {:0.0f} processes...".format(num_processes))
        t1 = time.time()
        window_kernels = window.Gaussian_Window_Kernels(k_centers, key)
        idx = range(len(k_centers))
        nBins = len(k_centers)

        p = Pool(processes=num_processes)
        WinFunAll=p.starmap(window_kernels.calc_gaussian_window_function, zip(idx, repeat(kmodes_sampled)))
        p.close()
        p.join()

        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = CovaPT_data_dir+'Wij_k'+str(nBins)+'_'+key+'.npy'
        b=np.zeros((len(idx),7,15,6))
        for i in range(len(idx)):
            b[i]=WinFunAll[i]
        np.save(save_file, b)
        print("window function saved to", save_file)

if __name__ == "__main__":
    main()