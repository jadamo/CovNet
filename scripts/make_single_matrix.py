# Simple script that uses CovaPT to create a single analytic covariance matrix given some 
# set of cosmology parameters
import time, os
import numpy as np

from CovNet import CovaPT

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def main():
    t1 = time.time()

    # define where to save the covariance matrix
    save_file = os.getcwd()+"/../data/Cov-test.npz"

    # option to display the resulting matrix with matplotlib
    plot_matrix = True

    # Define the cosmology parameters to use
    # [H0, omch2, ln(10^10 A_s), b1, b2, bG2, cs0, cs2, cbar, Pshot]
    params = np.array([67.77,0.1184,1, 1.9485,-0.5387, 0.1, 5, -15, 100., 5e3])
    # params = np.array([70.848,0.1120,0.7573, 2.8213,-0.2566, -0.0442, 12.0884, 4.54, 381.8, 984])
    #params = np.array([6.9383e+01, 1.18316e-01, 1.038e+00, 1.9094e+00, -2.956e+00, 2.06320e-01, 0, 0, 500, 0])
    
    # k bin centers to generate covariance for
    k = np.linspace(0.01, 0.19, 10)

    Analytic_Model = CovaPT.LSS_Model(0.61, k)
    C_G, C_SSC, C_T0 = Analytic_Model.get_full_covariance(params)
    C = C_G + C_SSC + C_T0

    t2 = time.time()
    print("Matrix generated in {:0.2f} s".format(t2 - t1))

    print("cond(C) = {:0.3e}".format(np.linalg.cond(C)))
    try:
        L = np.linalg.cholesky(C)
        print("Covariance matrix is positive-definite :)")
        
    except np.linalg.LinAlgError as err:
        print("Covariance matrix is NOT positive-definite!")

    np.savez(save_file, C_G=C_G, C_SSC=C_SSC, C_T0=C_T0)

    if plot_matrix:
        plt.figure()
        plt.imshow(C, cmap="RdBu", norm=colors.SymLogNorm(linthresh=1., vmin=np.amin(C), vmax=np.amax(C)))
        plt.colorbar()

if __name__ == "__main__":
    main()