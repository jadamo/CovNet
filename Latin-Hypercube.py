# This program takes bounds of our cosmological models and samples them via a 
# Latin hypercube with N samples, then saves those samples to file. Used to generate
# a training set on multiple machines
import numpy as np
from scipy.stats import qmc
import sys

def main():

    args = sys.argv[1:]
    N = int(args[0])

    # bounds either taken from Wadekar et al (2020) or assumed to be "very wide"
    # NOTE: H0, A, and ombh2 have no assumed priors in that paper, so I chose an arbitrary large range
    # NOTE: ns and omega_b have assumed values, as they claim using Planck priors makes no difference.
    # I'll therefore try to chose a range based on those priors found from https://wiki.cosmos.esa.int/planckpla/index.php/Cosmological_Parameters 
    # For As, the reference value is taken from https://arxiv.org/pdf/1807.06209.pdf table 1 (the best fit column), 
    # since Wadekar uses A = As / As_planck
    # ---Cosmology parameters sample bounds---
    H0_bounds    = [50, 100]      # Hubble constant
    omch2_bounds = [0.002, 0.3]   # Omega_cdm h^2
    ombh2_bounds = [0.005, 0.08]  # Omega b h^2
    As_planck = 3.0448
    As_bounds    = [0.2, 1.7]     # Ratio of Amplitude of Primordial Power spectrum (As / As_planck)
    ns_bounds    = [0.9, 1.1]     # Spectral index
    b1_bounds    = [1, 4]         # Linear bias       (b1 * (A/A_planck)^1/2)
    b2_bounds    = [-5, 5]        # non-linear bias?  (b2 * (A/A_planck)^1/2)

    # sample the distribution of points using a Latin Hypercube
    sampler = qmc.LatinHypercube(d=7)
    dist = sampler.random(n=N)

    # ---Cosmology parameters---
    #Omega_m = dist[:,0]*(Omega_m_bounds[1] - Omega_m_bounds[0]) + Omega_m_bounds[0]
    H0 = dist[:,0]*(H0_bounds[1] - H0_bounds[0]) + H0_bounds[0]
    omch2 = dist[:,1]*(omch2_bounds[1] - omch2_bounds[0]) + omch2_bounds[0]
    ombh2 = dist[:,2]*(ombh2_bounds[1] - ombh2_bounds[0]) + ombh2_bounds[0]
    As = dist[:,3]*(As_bounds[1] - As_bounds[0]) + As_bounds[0]
    ns = dist[:,4]*(ns_bounds[1] - ns_bounds[0]) + ns_bounds[0]
    b1 = dist[:,5]*(b1_bounds[1] - b1_bounds[0]) + b1_bounds[0]
    b2 = dist[:,6]*(b2_bounds[1] - b2_bounds[0]) + b2_bounds[0]

    header_str = "H0, omch2, ombh2, As, ns, b1 A^1/2, b2 A^1/2"
    data = np.vstack((H0, omch2, ombh2, As, ns, b1, b2)).T

    np.savetxt("Sample-params.txt", data, header=header_str)
    print("Saved " + str(N) + " sets of parameters")

if __name__ == "__main__":
    main()