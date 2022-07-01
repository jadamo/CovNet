# This program takes bounds of our cosmological models and samples them via a 
# Latin hypercube with N samples, then saves those samples to file. Used to generate
# a training set on multiple machines
import numpy as np
from scipy.stats import qmc
import sys

def main():

    args = sys.argv[1:]
    N = int(args[0])
    ranks = int(args[1])

    # bounds taken from table 8 of Wadekar et al 2020 (mean value +- 2x 1sigma interval)
    # For As, the reference value is taken from https://arxiv.org/pdf/1807.06209.pdf table 1, 
    # since Wadekar uses A = As / As_planck
    # ---Cosmology parameters sample bounds---
    #Omega_m_bounds = [0.2699, 0.3459]      # Omega matter
    H0_bounds      = [66.5, 75.5]          # Hubble constant
    #As_bounds  =     [8.586e-10, 2.031e-9] # Amplitude of Primordial Power spectrum <- double check this look at planck 1sigma range
    As_bounds  =     [2.4752, 3.7128] # Amplitude of Primordial Power spectrum <- double check this look at planck 1sigma range
    ombh2_bounds   = [0.0211375, 0.0233625]# Omega b h^2
    #omch2_bounds   = [0.1157, 0.1535]      # Omega_cdm h^2
    omch2_bounds   = [0.10782, 0.13178]      # Omega_cdm h^2
    b1_bounds      = [1.806, 2.04]         # Linear bias
    b2_bounds      = [-2.962, 0.458]       # non-linear bias?

    # sample the distribution of points using a Latin Hypercube
    sampler = qmc.LatinHypercube(d=6)
    dist = sampler.random(n=N)

    # ---Cosmology parameters---
    #Omega_m = dist[:,0]*(Omega_m_bounds[1] - Omega_m_bounds[0]) + Omega_m_bounds[0]
    H0 = dist[:,1]*(H0_bounds[1] - H0_bounds[0]) + H0_bounds[0]
    As = dist[:,2]*(As_bounds[1] - As_bounds[0]) + As_bounds[0]
    omch2 = dist[:,3]*(omch2_bounds[1] - omch2_bounds[0]) + omch2_bounds[0]
    ombh2 = dist[:,3]*(ombh2_bounds[1] - ombh2_bounds[0]) + ombh2_bounds[0]
    #ombh2=0.022  # Omega_b h^2 - this value is fixed
    b1 = dist[:,4]*(b1_bounds[1] - b1_bounds[0]) + b1_bounds[0]
    b2 = dist[:,5]*(b2_bounds[1] - b2_bounds[0]) + b2_bounds[0]

    header_str = "H0, As, omch2, ombh2, b1, b2"
    data = np.vstack((H0, As, omch2, ombh2, b1, b2)).T
    #print(data[0])
    #print(Omega_m[0], H0[0], As[0], omch2[0], b1[0], b2[0])
    np.savetxt("Sample-params.txt", data, header=header_str)
    print("Saved " + str(N) + " sets of parameters to " + str(ranks) + " files")

if __name__ == "__main__":
    main()