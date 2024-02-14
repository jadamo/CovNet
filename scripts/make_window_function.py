# This script calculates the window function used by the Gaussian covariance term in CovaPT
# NOTE: It is highly recommended you run this on multiple cpu cores, or on an hpc cluster
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import time
import numpy as np
from multiprocessing import Pool
from CovNet import window
from CovNet.config import CovaPT_data_dir

# how many processers to use
num_processes = 12

# number of kmodes to sample
# The default used by Jay Wadekar was 25000, which was run on a cluster
kmodes_sampled = 10

# K bins to generate the window function for
k_centers = np.linspace(0.01, 0.19, 10)

# String specifying what data chunk this window function was generated with
# NOTE: Currently only valid option is "HighZ_NGC"
key='HighZ_NGC'

def main():
    
    t1 = time.time()
    window_kernels = window.Window_Function(k_centers)
    idx = range(len(k_centers))
    nBins = len(k_centers)

    print("Starting window function generation with {:0.0f} processes...".format(num_processes))
    p = Pool(processes=num_processes)
    WinFunAll=p.map(window_kernels.WinFun, idx)
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