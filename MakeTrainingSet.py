# This script is basically CovaPT's jupyter notebook, but in script form so you can more easily run it

import scipy, time,sys, warnings, math
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import exp, log, log10, cos, sin, pi, cosh, sinh , sqrt, amin, amax, mean, dot, power, conj
from scipy.misc import derivative
import camb
from camb import model, initialpower
#from ctypes import c_double, c_int, cdll, byref
from multiprocessing import Pool
from itertools import repeat
from mpi4py import MPI

sys.path.insert(0, '/home/joeadamo/Research/CovaPT/detail')
import T0

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

dire='/home/joeadamo/Research/CovaPT/Example-Data/'
home_dir = "/home/joeadamo/Research/CovA-NN-Emulator/PCA-Set/"

#Using the window kernels calculated from the survey random catalog as input
#See Survey_window_kernels.ipynb for the code to generate these window kernels using the Wij() function
# The survey window used throughout this code is BOSS NGC-highZ (0.5<z<0.75)
WijFile=np.load(dire+'Wij_k120_HighZ_NGC.npy')

# Include here the theory power spectrum best-fitted to the survey data
# Currently I'm using here the Patchy output to show the comparison with mock catalogs later
k=np.loadtxt(dire+'k_Patchy.dat'); kbins=len(k) #number of k-bins

# Number of matrices to make
N = 100
# Number of processors to use
N_PROC = 16

# The following parameters are calculated from the survey random catalog
# Using Iij convention in Eq.(3)
alpha = 0.02; 
i22 = 454.2155*alpha; i11 = 7367534.87288*alpha; i12 = 2825379.84558*alpha;
i10 = 23612072*alpha; i24 = 58.49444652*alpha; 
i14 = 756107.6916375*alpha; i34 = 8.993832235e-3*alpha;
i44 = 2.158444115e-6*alpha; i32 = 0.11702382*alpha;
i12oi22 = 2825379.84558/454.2155; #Effective shot noise

#-------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------

#-------------------------------------------------------------------
# For generating individual elements of the Gaussian covariance matrix
# see Survey_window_kernels.ipynb for further details where the same function is used
def Cij(kt, Wij, Pfit):
    temp=np.zeros((7,6));
    for i in range(-3,4):
        if(kt+i<0 or kt+i>=kbins):
            temp[i+3]=0.
            continue
        temp[i+3]=Wij[i+3,0]*Pfit[0][kt]*Pfit[0][kt+i]+\
        Wij[i+3,1]*Pfit[0][kt]*Pfit[2][kt+i]+\
        Wij[i+3,2]*Pfit[0][kt]*Pfit[4][kt+i]+\
        Wij[i+3,3]*Pfit[2][kt]*Pfit[0][kt+i]+\
        Wij[i+3,4]*Pfit[2][kt]*Pfit[2][kt+i]+\
        Wij[i+3,5]*Pfit[2][kt]*Pfit[4][kt+i]+\
        Wij[i+3,6]*Pfit[4][kt]*Pfit[0][kt+i]+\
        Wij[i+3,7]*Pfit[4][kt]*Pfit[2][kt+i]+\
        Wij[i+3,8]*Pfit[4][kt]*Pfit[4][kt+i]+\
        1.01*(Wij[i+3,9]*(Pfit[0][kt]+Pfit[0][kt+i])/2.+\
        Wij[i+3,10]*Pfit[2][kt]+Wij[i+3,11]*Pfit[4][kt]+\
        Wij[i+3,12]*Pfit[2][kt+i]+Wij[i+3,13]*Pfit[4][kt+i])+\
        1.01**2*Wij[i+3,14]
    return(temp)

#-------------------------------------------------------------------
# Growth factor for LCDM cosmology
def Dz(z,Om0):
    return(scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3)
                                /scipy.special.hyp2f1(1/3., 1, 11/6., 1-1/Om0)/(1+z))

#-------------------------------------------------------------------
# Growth rate for LCDM cosmology
def fgrowth(z,Om0):
    return(1. + 6*(Om0-1)*scipy.special.hyp2f1(4/3., 2, 17/6., (1-1/Om0)/(1+z)**3)
                  /( 11*Om0*(1+z)**3*scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3) ))

#-------------------------------------------------------------------
def Pk_lin(H0, ombh2, omch2, As, z):
    """
    Generates a linear initial power spectrum from CAMB
    """
    #get matter power spectra and sigma8 at the redshift we want
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=0.965, As=np.exp(As)/1e10)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[z], kmax=0.25)

    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    # k bins will be interpolated to what we want later, so it's "ok" if this isn't exact
    kh, z1, pk = results.get_matter_power_spectrum(minkh=0.0025, maxkh=0.25, npoints = 100)
    s8 = np.array(results.get_sigma8())
    
    pdata = np.vstack((kh, pk[0])).T
    return pdata, s8

#-------------------------------------------------------------------
def trispIntegrand(u12,k1,k2,Plin):
    return( (8*i44*(Plin(k1)**2*T0.e44o44_1(u12,k1,k2) + Plin(k2)**2*T0.e44o44_1(u12,k2,k1))
            +16*i44*Plin(k1)*Plin(k2)*T0.e44o44_2(u12,k1,k2)
             +8*i34*(Plin(k1)*T0.e34o44_2(u12,k1,k2)+Plin(k2)*T0.e34o44_2(u12,k2,k1))
            +2*i24*T0.e24o44(u12,k1,k2))
            *Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12)) )

#-------------------------------------------------------------------
# Returns the tree-level trispectrum as a function of multipoles and k1, k2
def trisp(l1,l2,k1,k2, Plin):
    T0.l1=l1; T0.l2=l2
    expr = i44*(Plin(k1)**2*Plin(k2)*T0.ez3(k1,k2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2,k1))\
           +8*i34*Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2)

    temp = (quad(trispIntegrand, -1, 1,args=(k1,k2,Plin))[0]/2. + expr)/i22**2
    return(temp)
trisp = np.vectorize(trisp)

#-------------------------------------------------------------------
# Using the Z12 kernel which is defined in Eq. (A9) (equations copy-pasted from Generating_T0_Z12_expressions.nb)
def Z12Kernel(l,mu,be,b1,b2,g2,dlnpk):
    if(l==0):
        exp=(7*b1**2*be*(70 + 42*be + (-35*(-3 + dlnpk) + 3*be*(28 + 13*be - 14*dlnpk - 5*be*dlnpk))*mu**2) + 
            b1*(35*(47 - 7*dlnpk) + be*(798 + 153*be - 98*dlnpk - 21*be*dlnpk + 
            4*(84 + be*(48 - 21*dlnpk) - 49*dlnpk)*mu**2)) + 
            98*(5*b2*(3 + be) + 4*g2*(-5 + be*(-2 + mu**2))))/(735.*b1**2)
    elif(l==2):
        exp=(14*b1**2*be*(14 + 12*be + (2*be*(12 + 7*be) - (1 + be)*(7 + 5*be)*dlnpk)*mu**2) + 
            b1*(4*be*(69 + 19*be) - 7*be*(2 + be)*dlnpk + 
            (24*be*(11 + 6*be) - 7*(21 + be*(22 + 9*be))*dlnpk)*mu**2 + 7*(-8 + 7*dlnpk + 24*mu**2)) + 
            28*(7*b2*be + g2*(-7 - 13*be + (21 + 11*be)*mu**2)))/(147.*b1**2)
    elif(l==4):
        exp=(8*be*(b1*(-132 + 77*dlnpk + be*(23 + 154*b1 + 14*dlnpk)) - 154*g2 + 
            (b1*(396 - 231*dlnpk + be*(272 + 308*b1 + 343*b1*be - 7*(17 + b1*(22 + 15*be))*dlnpk)) + 
            462*g2)*mu**2))/(2695.*b1**2)
    return(exp)

#-------------------------------------------------------------------
# Legendre polynomials
def lp(l,mu):
    if (l==0): exp=1
    if (l==2): exp=((3*mu**2 - 1)/2.)
    if (l==4): exp=((35*mu**4 - 30*mu**2 + 3)/8.)
    return(exp)

#-------------------------------------------------------------------
# For transforming the linear array to a matrix
def MatrixForm(a):
    b=np.zeros((3,3))
    if len(a)==6:
        b[0,0]=a[0]; b[1,0]=b[0,1]=a[1]; 
        b[2,0]=b[0,2]=a[2]; b[1,1]=a[3];
        b[2,1]=b[1,2]=a[4]; b[2,2]=a[5];
    if len(a)==9:
        b[0,0]=a[0]; b[0,1]=a[1]; b[0,2]=a[2]; 
        b[1,0]=a[3]; b[1,1]=a[4]; b[1,2]=a[5];
        b[2,0]=a[6]; b[2,1]=a[7]; b[2,2]=a[8];
    return(b)

#-------------------------------------------------------------------
# Calculating multipoles of the Z12 kernel
def Z12Multipoles(i,l,be,b1,b2,g2,dlnpk):
    return(quad(lambda mu: lp(i,mu)*Z12Kernel(l,mu,be,b1,b2,g2,dlnpk), -1, 1)[0])
Z12Multipoles = np.vectorize(Z12Multipoles)

#-------------------------------------------------------------------
def CovLATerm(sigma22x10, dlnPk, be,b1,b2,g2):
    """
    Calculates the LA terms used in the SSC calculations
    """
    covaLAterm=np.zeros((3,len(k)))
    for l in range(3):
        for i in range(3):
            for j in range(3):
                covaLAterm[l]+=1/4.*sigma22x10[i,j]*Z12Multipoles(2*i,2*l,be,b1,b2,g2,dlnPk)\
                *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1)[0]
    return covaLAterm
        
#-------------------------------------------------------------------
def covaSSC(l1,l2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk):
    """
    Returns the SSC covariance matrix contrbution
    BC= beat-coupling, LA= local average effect, SSC= super sample covariance
    """
    
    covaBC=np.zeros((len(k),len(k)))
    for i in range(3):
        for j in range(3):
            covaBC+=1/4.*sigma22Sq[i,j]*np.outer(Plin(k)*Z12Multipoles(2*i,l1,be,b1,b2,g2,dlnPk),Plin(k)*Z12Multipoles(2*j,l2,be,b1,b2,g2,dlnPk))
            sigma10Sq[i,j]=1/4.*sigma10Sq[i,j]*quad(lambda mu: lp(2*i,mu)*(1 + be*mu**2), -1, 1)[0]\
            *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1)[0]

    covaLA=-rsd[l2]*np.outer(Plin(k)*(covaLAterm[int(l1/2)]+i32/i22/i10*rsd[l1]*Plin(k)*b2/b1**2+2/i10*rsd[l1]),Plin(k))\
           -rsd[l1]*np.outer(Plin(k),Plin(k)*(covaLAterm[int(l2/2)]+i32/i22/i10*rsd[l2]*Plin(k)*b2/b1**2+2/i10*rsd[l2]))\
           +(np.sum(sigma10Sq)+1/i10)*rsd[l1]*rsd[l2]*np.outer(Plin(k),Plin(k))

    return(covaBC+covaLA)
    
#-------------------------------------------------------------------
def CovMatGauss(Pfit):
    """
    Returns the full (Monopole+Quadrupole) Gaussian covariance matrix
    """
    covMat=np.zeros((2*kbins,2*kbins))
    for i in range(kbins):
        temp=Cij(i,WijFile[i], Pfit)
        C00=temp[:,0]; C22=temp[:,1]; C20=temp[:,3]
        for j in range(-3,4):
            if(i+j>=0 and i+j<kbins):
                covMat[i,i+j]=C00[j+3]
                covMat[kbins+i,kbins+i+j]=C22[j+3]
                covMat[kbins+i,i+j]=C20[j+3]
    covMat[:kbins,kbins:kbins*2]=np.transpose(covMat[kbins:kbins*2,:kbins])
    covMat=(covMat+np.transpose(covMat))/2.
    return(covMat)

#-------------------------------------------------------------------
def CovMatNonGauss(Plin, be,b1,b2,g2):
    """
    Returns the full Non-Gaussian covariance matrix
    """
    # Get the derivativee of the linear power spectrum
    dlnPk=derivative(Plin,k,dx=1e-4)*k/Plin(k)
    
    # Loading window power spectra calculated from the survey random catalog (code will be uploaded in a different notebook)
    # These are needed to calculate the sigma^2 terms
    # Columns are k P00 P02 P04 P22 P24 P44 Nmodes
    powW22=np.loadtxt(dire+'WindowPower_W22_highz.dat')
    powW10=np.loadtxt(dire+'WindowPower_W10_highz.dat')

    # Columns are k P00 P02 P04 P20 P22 P24 P40 P42 P44 Nmodes
    powW22x10=np.loadtxt(dire+'WindowPower_W22xW10_highz.dat')
    
    # Kaiser terms
    rsd=np.zeros(5)
    rsd[0]=1 + (2*be)/3 + be**2/5
    rsd[2]=(4*be)/3 + (4*be**2)/7
    rsd[4]=(8*be**2)/35
    
    # Calculating the RMS fluctuations of supersurvey modes 
    #(e.g., sigma22Sq which was defined in Eq. (33) and later calculated in Eq.(65)
    kwin = powW22[:,0]
    [temp,temp2]=np.zeros((2,6)); temp3 = np.zeros(9)
    for i in range(9):
        Pwin=InterpolatedUnivariateSpline(kwin, powW22x10[:,1+i])
        temp3[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1])[0]

        if(i<6):
            Pwin=InterpolatedUnivariateSpline(kwin, powW22[:,1+i])
            temp[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1])[0]
            Pwin=InterpolatedUnivariateSpline(kwin, powW10[:,1+i])
            temp2[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1])[0]
        else:
            continue
    
    sigma22Sq = MatrixForm(temp); sigma10Sq = MatrixForm(temp2); sigma22x10 = MatrixForm(temp3)
  
    # Calculate the LA term
    covaLAterm = CovLATerm(sigma22x10, dlnPk, be,b1,b2,g2)
    
    # Calculate the Non-Gaussian multipole covariance
    # Warning: the trispectrum takes a while to run
    covaT0mult=np.zeros((2*kbins,2*kbins))
    for i in range(len(k)):
        covaT0mult[i,:kbins]=trisp(0,0,k[i],k, Plin)
        covaT0mult[i,kbins:]=trisp(0,2,k[i],k, Plin)
        covaT0mult[kbins+i,kbins:]=trisp(2,2,k[i],k, Plin)

    covaT0mult[kbins:,:kbins]=np.transpose(covaT0mult[:kbins,kbins:])

    covaSSCmult=np.zeros((2*kbins,2*kbins))
    covaSSCmult[:kbins,:kbins]=covaSSC(0,0, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    covaSSCmult[kbins:,kbins:]=covaSSC(2,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    covaSSCmult[:kbins,kbins:]=covaSSC(0,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk); 
    covaSSCmult[kbins:,:kbins]=np.transpose(covaSSCmult[:kbins,kbins:])

    covaNG=covaT0mult+covaSSCmult
    return covaNG

def CovAnalytic(H0, Pfit, Omega_m, ombh2, omch2, As, z, b1, b2, b3, be, g2, g3, g2x, g21, i):
    """
    Generates and returns the full analytic covariance matrix. This function is meant to be run
    in parallel.
    Also returns sigma8, which is derived when calculating the initial power spectrum
    """
    # initializing bias parameters for trispectrum
    T0.InitParameters([b1,be,g2,b2,g3,g2x,g21,b3])

    #num_matrices = 0; tmin = 1e10; tmax = 0
    #while t2 - t1 < 60*60*24:
    # Get initial power spectrum
    pdata, s8 = Pk_lin(H0, ombh2, omch2, As, z)
    Plin=InterpolatedUnivariateSpline(pdata[:,0], Dz(z, Omega_m)**2*b1**2*pdata[:,1])

    # Calculate the covariance
    covaG  = CovMatGauss(Pfit)
    covaNG = CovMatNonGauss(Plin, be,b1,b2,g2)

    covAnl=covaG+covaNG

    # save results to a file for training
    #header_str = "H0, Omega_m, omch2, As, sigma8, b1, b2\n"
    #header_str += str(H0) + ", " + str(Omega_m) + ", " + str(omch2) + ", " + str(As) + ", " + str(s8[0]) + ", " + str(b1) + ", " + str(b2)
    idx = f'{i:04d}'
    params = np.array([H0, Omega_m, omch2, ombh2, As, b1, b2])
    #np.savetxt(home_dir+"Training-Set/CovA-"+idx+".txt", covAnl, header=header_str)
    np.savez(home_dir+"CovA-"+idx+".npz", params=params, C=covAnl)
    #return covAnl

#-------------------------------------------------------------------
# MAIN
#-------------------------------------------------------------------
def main():
    
    comm = MPI.COMM_WORLD
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    # TEMP: ignore integration warnings to make output more clean
    warnings.filterwarnings("ignore")
    
    Pfit=[0,0,0,0,0]
    Pfit[0]=np.loadtxt(dire+'P0_fit_Patchy.dat')
    Pfit[2]=np.loadtxt(dire+'P2_fit_Patchy.dat')
    Pfit[4]=np.loadtxt(dire+'P4_fit_Patchy.dat')
    t1 = time.time(); t2 = t1

    # Split up data to multiple MPI ranks
    # Aparently MPI scatter doesn't work on Puma, so this uses a different way
    assert N % size == 0
    offset = int((N / size) * rank)
    data_len = int(N / size)
    data = np.loadtxt("Sample-params-PCA.txt", skiprows=1+offset, max_rows = data_len)
    # send_chunk = None
    # if rank == 0:
    #     send_data = np.loadtxt("Sample-params.txt", skiprows=1)
    #     send_chunk = np.array_split(send_data, size, axis=0)
    # data = np.empty((int(N/size),6), dtype=np.float64)
    # data = comm.scatter(send_chunk, root=0)
    #comm.Scatter([send_chunk, MPI.DOUBLE], [data, MPI.DOUBLE], root=0)

    # ---Cosmology parameters---
    #Omega_m = data[:,0]
    H0 = data[:,0]
    As = data[:,1]
    omch2 = data[:,2]
    ombh2 = data[:,3]
    b1 = data[:,4]
    b2 = data[:,5]

    Omega_m = (omch2 + ombh2) / (H0/100)**2
    # Below are expressions for non-local bias (g_i) from local lagrangian approximation
    # and non-linear bias (b_i) from peak-background split fit of 
    # Lazyeras et al. 2016 (rescaled using Appendix C.2 of arXiv:1812.03208),
    # which could used if those parameters aren't constrained.
    g2 = -2/7*(b1 - 1)
    g3 = 11/63*(b1 - 1)
    #b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2 
    g2x = -2/7*b2
    g21 = -22/147*(b1 - 1)
    b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x - 4/3*g3 - 8/3*g21 - 32/21*g2
    
    # ---Bias and survey parameters---
    z = 0.58 #mean redshift of the high-Z chunk
    be = fgrowth(z, Omega_m)/b1; #beta = f/b1, zero for real space
    
    # split up workload to different nodes
    i = np.arange(offset, offset+data_len, dtype=np.int)
    # send_i = np.empty(N, dtype=np.int)
    # if rank == 0:
    #     send_i = np.arange(N, dtype=np.int)
    #     send_chunk = np.array_split(send_i, size, axis=0)
    # i = np.empty(int(N / size), dtype=np.int)
    # i = comm.scatter(send_chunk, root=0)

    # initialize pool for multiprocessing
    t1 = time.time()
    with Pool(processes=N_PROC) as pool:
        pool.starmap(CovAnalytic, zip(H0, repeat(Pfit), Omega_m, repeat(ombh2), omch2, As, 
                                               repeat(z), b1, b2, b3, be, b2, g3, g2x, g21, i))
        #(H0, Pfit, Omega_m, ombh2, omch2, z, b1, b2, b3, be, g2, g3, g2x, g21)
    t2 = time.time()
    
    # save this matrix to a file
    print("Done! Took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))
    print("Made " + str(N) + " matrices")
    
if __name__ == "__main__":
    main()