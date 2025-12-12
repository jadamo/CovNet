# Simple script that uses CovaPT to create a single analytic covariance matrix given some 
# set of cosmology parameters
import time, os
import numpy as np

from CovNet import CovaPT
from CovNet import window
from multiprocessing import Pool
from itertools import repeat

import matplotlib.pyplot as plt
import matplotlib.colors as colors

num_processes = 8

# number of kmodes to sample
# The default used by Jay Wadekar was 25000, which was run on a cluster
kmodes_sampled = 500

box_pad = 1.2

z_bounds = [1.0, 1.6]

start_from_scratch = True

key = "spherex"

# define where to save the covariance matrix
save_dir = "/home/joeadamo/Research/Data/thecov_data/covnet_ng_lin/"

# option to display the resulting matrix with matplotlib
plot_matrix = False

def calc_ffts(survey_kernels:window.Survey_Window_Kernels, data_dir, key):
    print("\nStarting FFT calculations...")
    t1 = time.time()
    export = survey_kernels.calc_gaussian_kernels(48, 3750)
    t2 = time.time()
    print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    save_file = os.path.join(data_dir,'FFTWinFun_'+key+'.npy')
    np.save(save_file, export)
    print("FFTs saved to", save_file)
    
def calc_SSC_window(survey_kernels:window.Survey_Window_Kernels, data_dir, key):
    print("\nStarting FFT calculations...")
    t1 = time.time()
    P_W = survey_kernels.calc_SSC_window_function(200, 7500)
    t2 = time.time()
    print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    save_file = os.path.join(data_dir,'WindowPowers_'+key+'.npy')
    np.save(save_file,P_W)
    print("SSC window functions saved to", save_file)

def calc_gaussian_window(window_kernels:window.Gaussian_Window_Kernels, data_dir, key, k_centers):
    print("\nStarting Gaussian window function generation with {:0.0f} processes...".format(num_processes))
    t1 = time.time()
    idx = range(len(k_centers))
    nBins = len(k_centers)

    window_kernels.get_shell_modes()
    p = Pool(processes=num_processes)
    WinFunAll=p.starmap(window_kernels.calc_gaussian_window_function, zip(idx, repeat(kmodes_sampled)))
    p.close()
    p.join()

    t2 = time.time()
    print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    save_file = os.path.join(data_dir, 'Wij_k'+str(nBins)+'_'+key+'.npy')
    b=np.zeros((len(idx),7,15,6))
    for i in range(len(idx)):
        b[i]=WinFunAll[i]
    np.save(save_file, b)
    print("window function saved to", save_file)

def main():

    #logging.basicConfig(level = logging.INFO)
    print("Starting from scratch:", start_from_scratch)

    # Define the cosmology parameters to use
    # [H0, omch2, A_s, b1, b2, bG2, cs0, cs2, cbar, Pshot]
    params = np.array([67.36, 0.1201, 2.1e-9, 2.2, 0.279, -0.6857, 0., 0., 0., 1./0.000796])

    # k bin centers to generate covariance for
    k = np.load(os.path.join(save_dir, "lin_karray.npy"))
    print(f"k_centers = {k}")

    # load in pre-computed data vector
    galaxy_ps = np.load(os.path.join(save_dir, "ps_1loop_hubble_true_link_galaxy_ps.npy"))
    galaxy_ps = [galaxy_ps[0,0,:,0], np.zeros(len(k)), galaxy_ps[0,0,:,1], np.zeros(len(k)), np.zeros(len(k))]

    survey_kernels = window.Survey_Window_Kernels(0.7, 0.31, "spherex", save_dir, z_bounds)
    print(f"I22 = {survey_kernels.I22}")
    print(f"box size = {survey_kernels.box_size}")

    if not os.path.exists(os.path.join(save_dir,'FFTWinFun_'+key+'.npy')) or start_from_scratch:
        t1 = time.time()
        calc_ffts(survey_kernels, save_dir, key)
        print("FFTs generated in {:0.2f} s".format(time.time() - t1))

    if not os.path.exists(os.path.join(save_dir,'WindowPowers_'+key+'.npy')) or start_from_scratch:
        t1 = time.time()
        calc_SSC_window(survey_kernels, save_dir, key)
        print("SSC window generated in {:0.2f} s".format(time.time() - t1))

    if not os.path.exists(os.path.join(save_dir, 'Wij_k'+str(len(k))+'_'+key+'.npy')) or start_from_scratch:
        t1 = time.time()
        window_kernels = window.Gaussian_Window_Kernels(k, key, data_dir=save_dir, sampling_mode="linear", 
                                                        lbox=survey_kernels.box_size * box_pad)
        window_kernels.set_I22(survey_kernels.I22)
        calc_gaussian_window(window_kernels, save_dir, key, k)
        print("Gaussian window generated in {:0.2f} s".format(time.time() - t1))

    t1 = time.time()
    Analytic_Model = CovaPT.LSS_Model(1.3, k, alpha=0.4, key=key, window_dir=save_dir)
    Analytic_Model.set_normalizations(survey_kernels.randoms)
    C_G, C_NG = Analytic_Model.get_full_covariance(params, Pk_galaxy=galaxy_ps, seperate_terms=True)
    C = C_G + C_NG

    t2 = time.time()
    print("Matrix generated in {:0.2f} s".format(t2 - t1))

    print("cond(C) = {:0.3e}".format(np.linalg.cond(C_G + C_NG)))
    try:
        L = np.linalg.cholesky(C_G)
        print("Gaussian term is positive-definite :)")
        
    except np.linalg.LinAlgError as err:
        print("Gaussian term is NOT positive-definite!")
    try:
        L = np.linalg.cholesky(C)
        print("Full covariance matrix is positive-definite :)")
        
    except np.linalg.LinAlgError as err:
        print("Full covariance matrix is NOT positive-definite!")

    np.savez(os.path.join(save_dir, "cov_covnet.npz"), C_G=C_G, C_NG=C_NG)

    if plot_matrix:
        plt.figure()
        plt.title("Gaussian term")
        min_val = 1e-15
        max_val = 1e15
        plt.imshow(C_G, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=min_val, vmax=max_val))
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])


        plt.figure()
        plt.title("T0 + SSC terms")
        min_val = 1e-15
        max_val = 1e15
        plt.imshow(C_NG, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=min_val, vmax=max_val))
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.show()

if __name__ == "__main__":
    main()