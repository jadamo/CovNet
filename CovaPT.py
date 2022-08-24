# This file is simply a repackaging of the functions and math found in CovaPT
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT
# NOTE: CovaPT must be downloaded for these functions to work

import scipy, sys
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import pi
from scipy.misc import derivative
import camb
from camb import model
from classy import Class
#sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_hod

#sys.path.insert(0, '/home/u12/jadamo/CovaPT/detail')
sys.path.insert(0, '/home/joeadamo/Research/CovaPT/detail')
import T0

#-------------------------------------------------------------------
# GLOBAL VARIABLES
#-------------------------------------------------------------------

# directory with window functions (I assumed these are calculated beforehand)
dire='/home/joeadamo/Research/CovaPT/Example-Data/'

#Using the window kernels calculated from the survey random catalog as input
#See Survey_window_kernels.ipynb for the code to generate these window kernels using the Wij() function
# The survey window used throughout this code is BOSS NGC-highZ (0.5<z<0.75)
WijFile=np.load(dire+'Wij_k120_HighZ_NGC.npy')

# Include here the theory power spectrum best-fitted to the survey data
# Currently I'm using here the Patchy output to show the comparison with mock catalogs later
k=np.loadtxt(dire+'k_Patchy.dat'); kbins=len(k) #number of k-bins

# Loading window power spectra calculated from the survey random catalog (code will be uploaded in a different notebook)
# These are needed to calculate the sigma^2 terms
# Columns are k P00 P02 P04 P22 P24 P44 Nmodes
powW22=np.loadtxt(dire+'WindowPower_W22_highz.dat')
powW10=np.loadtxt(dire+'WindowPower_W10_highz.dat')

# Columns are k P00 P02 P04 P20 P22 P24 P40 P42 P44 Nmodes
powW22x10=np.loadtxt(dire+'WindowPower_W22xW10_highz.dat')

# The following parameters are calculated from the survey random catalog
# Using Iij convention in Eq.(3)
alpha = 0.02; 
i22 = 454.2155*alpha; i11 = 7367534.87288*alpha; i12 = 2825379.84558*alpha;
i10 = 23612072*alpha; i24 = 58.49444652*alpha; 
i14 = 756107.6916375*alpha; i34 = 8.993832235e-3*alpha;
i44 = 2.158444115e-6*alpha; i32 = 0.11702382*alpha;
i12oi22 = 2825379.84558/454.2155; #Effective shot noise

# Galaxy parameters for generating galaxy power spectra
gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}

#-------------------------------------------------------------------
# HELPER FUNCTIONS
#-------------------------------------------------------------------

#-------------------------------------------------------------------
# For generating individual elements of the Gaussian covariance matrix
# see Survey_window_kernels.ipynb for further details where the same function is used
def Cij(kt, Wij, Pfit):
    temp=np.zeros((7,6))
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

    temp = (quad(trispIntegrand, -1, 1,args=(k1,k2,Plin), limit=60)[0]/2. + expr)/i22**2
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
    return(quad(lambda mu: lp(i,mu)*Z12Kernel(l,mu,be,b1,b2,g2,dlnpk), -1, 1, limit=60)[0])
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
                *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]
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
            sigma10Sq[i,j]=1/4.*sigma10Sq[i,j]*quad(lambda mu: lp(2*i,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]\
            *quad(lambda mu: lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]

    covaLA=-rsd[l2]*np.outer(Plin(k)*(covaLAterm[int(l1/2)]+i32/i22/i10*rsd[l1]*Plin(k)*b2/b1**2+2/i10*rsd[l1]),Plin(k))\
           -rsd[l1]*np.outer(Plin(k),Plin(k)*(covaLAterm[int(l2/2)]+i32/i22/i10*rsd[l2]*Plin(k)*b2/b1**2+2/i10*rsd[l2]))\
           +(np.sum(sigma10Sq)+1/i10)*rsd[l1]*rsd[l2]*np.outer(Plin(k),Plin(k))

    return(covaBC+covaLA)

#-------------------------------------------------------------------
# Functions meant to be called elsewhere
# ------------------------------------------------------------------

#-------------------------------------------------------------------
def Pk_gg(params, pgg):
    """
    Calculates the galaxy power spectrum using Yosuke's dark emulator
    """
    h, omch2, ombh2, As = params[0] / 100, params[1], params[2], params[3]
    ns = 0.965
    Om0 = (omch2 + ombh2 + 0.00064) / (h**2)
    
    # rebuild parameters into correct format (ombh2, omch2, 1-Om0, ln As, ns, w)
    cparams = np.array([ombh2, omch2, 1-Om0, As, ns, -1])
    redshift = 0.58
    k = np.linspace(0.005, 0.25, 50)
    #mu = np.linspace(0.1,0.9,4)
    alpha_perp = 1.1
    alpha_para = 1

    pgg.set_cosmology(cparams, redshift) # <- takes ~0.17s to run
    pgg.set_galaxy(gparams)
    # takes ~0.28 s to run
    P0_emu = pgg.get_pl_gg_ref(0, k, alpha_perp, alpha_para, name='total')
    P2_emu = pgg.get_pl_gg_ref(2, k, alpha_perp, alpha_para, name='total')
    P4_emu = pgg.get_pl_gg_ref(4, k, alpha_perp, alpha_para, name='total')
    return [P0_emu, 0, P2_emu, 0, P4_emu]

#-------------------------------------------------------------------
def Pk_CLASS_PT(params):
    z = 0.5
    cosmo = Class()
    cosmo.set({'output':'mPk',
            'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes', # use CDM+baryon spectra
            'RSD':'Yes',
            'AP':'Yes', # Alcock-Paczynski effect
            'Omfid':'0.31', # fiducial Omega_m
            'PNG':'No', # single-field inflation PNG
            'FFTLog mode':'FAST',
            'A_s':np.exp(params[3])/1e10,
            'n_s':0.9649,
            'tau_reio':0.052,
            'omega_b':params[2],
            'omega_cdm':params[1],
            'h':params[0] / 100.,
            'YHe':0.2425,
            'N_ur':2.0328,
            'N_ncdm':1,
            'm_ncdm':0.06,
            'z_pk':z
            })
    k = np.linspace(0.005, 0.25, 50)
    cosmo.compute()
    cosmo.initialize_output(k, z, len(k))

    b1, b2, bG2, bGamma3, cs0, cs2, cs4, Pshot, b4 = params[4], params[5], 0.1, -0.1, 0., 30., 0., 3000., 10.
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)
    pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, b4)
    # Ths line is necesary to prevent memory leaks
    cosmo.struct_cleanup()
    return [pk_g0, 0, pk_g2, 0, pk_g4]

#-------------------------------------------------------------------
def get_gaussian_covariance(params, pgg=None, Pk_galaxy=None):
    """
    Returns the (Monopole+Quadrupole) Gaussian covariance matrix
    If Pk_galaxy is already calculated, takes ~10 ms to run
    Else, takes ~0.5 s
    """
    # generate galaxy redshift-space power spectrum if necesary
    if Pk_galaxy == None:
        if pgg == None: pgg = pkmu_hod()
        H0, omch2, ombh2, As = params[0], params[1], params[2], params[3]
        Pk_galaxy = Pk_gg(H0, omch2, ombh2, As, pgg)

    covMat=np.zeros((2*kbins,2*kbins))
    for i in range(kbins):
        temp=Cij(i,WijFile[i], Pk_galaxy)
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
def get_non_gaussian_covariance(params):
    """
    Returns the Non-Gaussian portion of the covariance matrix
    Takes ~ 10 minutes to run
    """
    # unpack parameters
    H0, omch2, ombh2, As, b1, b2 = params[0], params[1], params[2], params[3], params[4], params[5]
    Omega_m = (omch2 + ombh2 + 0.00064) / (H0/100)**2
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

    # initializing bias parameters for trispectrum
    T0.InitParameters([b1,be,g2,b2,g3,g2x,g21,b3])

    # Get initial power spectrum
    pdata, s8 = Pk_lin(H0, ombh2, omch2, As, z)
    Plin=InterpolatedUnivariateSpline(pdata[:,0], Dz(z, Omega_m)**2*b1**2*pdata[:,1])

    # Get the derivativee of the linear power spectrum
    dlnPk=derivative(Plin,k,dx=1e-4)*k/Plin(k)
    
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
        temp3[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]

        if(i<6):
            Pwin=InterpolatedUnivariateSpline(kwin, powW22[:,1+i])
            temp[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]
            Pwin=InterpolatedUnivariateSpline(kwin, powW10[:,1+i])
            temp2[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]
        else:
            continue
    
    sigma22Sq = MatrixForm(temp); sigma10Sq = MatrixForm(temp2); sigma22x10 = MatrixForm(temp3)
  
    # Calculate the LA term
    covaLAterm = CovLATerm(sigma22x10, dlnPk, be,b1,b2,g2)
    
    covaSSCmult=np.zeros((2*kbins,2*kbins))
    covaSSCmult[:kbins,:kbins]=covaSSC(0,0, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    covaSSCmult[kbins:,kbins:]=covaSSC(2,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    covaSSCmult[:kbins,kbins:]=covaSSC(0,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk); 
    covaSSCmult[kbins:,:kbins]=np.transpose(covaSSCmult[:kbins,kbins:])

    # Calculate the Non-Gaussian multipole covariance
    # Warning: the trispectrum takes a while to run
    covaT0mult=np.zeros((2*kbins,2*kbins))
    for i in range(len(k)):
        covaT0mult[i,:kbins]=trisp(0,0,k[i],k, Plin)
        covaT0mult[i,kbins:]=trisp(0,2,k[i],k, Plin)
        covaT0mult[kbins+i,kbins:]=trisp(2,2,k[i],k, Plin)

    covaT0mult[kbins:,:kbins]=np.transpose(covaT0mult[:kbins,kbins:])

    covaNG=covaT0mult+covaSSCmult
    #return covaNG
    return covaSSCmult, covaT0mult

#-------------------------------------------------------------------
def get_full_covariance(params, pgg, Pk_galaxy=None):
    """
    Returns the full analytic covariance matrix
    """
    cov_G = get_gaussian_covariance(params, pgg, Pk_galaxy)
    cov_NG = get_non_gaussian_covariance(params)
    return cov_G + cov_NG