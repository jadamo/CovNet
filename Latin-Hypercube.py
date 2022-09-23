# This program takes bounds of our cosmological models and samples them via a 
# Latin hypercube with N samples, then saves those samples to file. Used to generate
# a training set on multiple machines
import numpy as np
from scipy.stats import qmc
import sys

def main():

    args = sys.argv[1:]
    N = int(args[0])

    # bounds taken from table 8 of Wadekar et al 2020 (mean value +- 3x 1sigma interval)
    # For As, the reference value is taken from https://arxiv.org/pdf/1807.06209.pdf table 1, 
    # since Wadekar uses A = As / As_planck
    # ---Cosmology parameters sample bounds---
    H0_bounds    = [60, 75]       # Hubble constant
    As_bounds    = [2.4, 3.8]     # Amplitude of Primordial Power spectrum <- double check this look at planck 1sigma range
    ns_bounds    = [0.94, 0.99]   # Spectral index
    ombh2_bounds = [0.02, 0.025]  # Omega b h^2
    omch2_bounds = [0.09, 0.15]   # Omega_cdm h^2
    b1_bounds    = [1.7, 2.45]    # Linear bias
    b2_bounds    = [-3.5, 0.75]   # non-linear bias?

    # sample the distribution of points using a Latin Hypercube
    sampler = qmc.LatinHypercube(d=7)
    dist = sampler.random(n=N)

    # ---Cosmology parameters---
    #Omega_m = dist[:,0]*(Omega_m_bounds[1] - Omega_m_bounds[0]) + Omega_m_bounds[0]
    H0 = dist[:,0]*(H0_bounds[1] - H0_bounds[0]) + H0_bounds[0]
    As = dist[:,1]*(As_bounds[1] - As_bounds[0]) + As_bounds[0]
    ns = dist[:,2]*(ns_bounds[1] - ns_bounds[0]) + ns_bounds[0]
    omch2 = dist[:,3]*(omch2_bounds[1] - omch2_bounds[0]) + omch2_bounds[0]
    ombh2 = dist[:,4]*(ombh2_bounds[1] - ombh2_bounds[0]) + ombh2_bounds[0]
    b1 = dist[:,5]*(b1_bounds[1] - b1_bounds[0]) + b1_bounds[0]
    b2 = dist[:,6]*(b2_bounds[1] - b2_bounds[0]) + b2_bounds[0]

    header_str = "H0, As, ns, omch2, ombh2, b1, b2"
    data = np.vstack((H0, As, ns, omch2, ombh2, b1, b2)).T

    np.savetxt("Sample-params.txt", data, header=header_str)
    print("Saved " + str(N) + " sets of parameters")

if __name__ == "__main__":
    main()