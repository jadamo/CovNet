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

#sys.path.insert(0, '/home/u12/jadamo/CovaPT/detail')
sys.path.insert(0, '/home/joeadamo/Research/CovaPT/detail')
#sys.path.insert(0, '/home/jadamo/UArizona/Research/CovaPT/detail')
import T0

# Organize everything into a class to more clearly specify things like
# what k range and binning we're using
class Analytic_Covmat():

    def __init__(self, z, k=np.linspace(0.005, 0.245, 25)):
        self.k = k
        self.kbins=len(k)
        self.z = z

        # directory with window functions (I assumed these are calculated beforehand)
        #self.dire='/home/u12/jadamo/CovaPT/Example-Data/'
        self.dire='/home/joeadamo/Research/CovaPT/Example-Data/Local/'
        #self.dire='/home/jadamo/UArizona/Research/CovaPT/Example-Data/'

        #Using the window kernels calculated from the survey random catalog as input
        #See Survey_window_kernels.ipynb for the code to generate these window kernels using the Wij() function
        try:
            self.WijFile = np.load(self.dire+'/Wij_k'+str(self.kbins)+'_HighZ_NGC.npy')
        except IOError:
            print("ERROR! Incorrect window kernel size! Please double-check your path or recalculate with the correct binning using Survey_window_kernels.ipynb")
            return -1

        # A, ns, ombh2 from Planck best-fit
        self.A_planck = 3.0447
        self.ns_planck = 0.9649
        self.ombh2_planck = 0.02237

        # Loading window power spectra calculated from the survey random catalog (code will be uploaded in a different notebook)
        # These are needed to calculate the sigma^2 terms
        # Columns are k P00 P02 P04 P22 P24 P44 Nmodes
        self.powW22=np.loadtxt(self.dire+'WindowPower_W22_highz.dat')
        self.powW10=np.loadtxt(self.dire+'WindowPower_W10_highz.dat')

        # Columns are k P00 P02 P04 P20 P22 P24 P40 P42 P44 Nmodes
        self.powW22x10=np.loadtxt(self.dire+'WindowPower_W22xW10_highz.dat')

        # The following parameters are calculated from the survey random catalog
        # Using Iij convention in Eq.(3)
        alpha = 0.02; 
        self.i22 = 454.2155*alpha; self.i11 = 7367534.87288*alpha; self.i12 = 2825379.84558*alpha;
        self.i10 = 23612072*alpha; self.i24 = 58.49444652*alpha; 
        self.i14 = 756107.6916375*alpha; self.i34 = 8.993832235e-3*alpha;
        self.i44 = 2.158444115e-6*alpha; self.i32 = 0.11702382*alpha;
        self.i12oi22 = 2825379.84558/454.2155; #Effective shot noise

        # vectorize some of the more expensive functions
        self.vec_Z12Multipoles = np.vectorize(self.Z12Multipoles)
        self.vec_trisp = np.vectorize(self.trisp)

        # k bins for the theory power spectrum
        dkf = 0.001
        self.kbins3 = np.zeros(400)
        for i in range(400): self.kbins3[i] = dkf/2+i*dkf

        # common CLASS-PT settings
        self.common_settings = {'output':'mPk',         # what to output
                            'non linear':'PT',      # {None, Halofit, PT}
                            'IR resummation':'Yes',
                            'Bias tracers':'Yes',
                            'cb':'Yes',             # use CDM+baryon spectra
                            'RSD':'Yes',            # Redshift-space distortions
                            'AP':'Yes',             # Alcock-Paczynski effect
                            'Omfid':'0.31',         # fiducial Omega_m
                            #'PNG':'No',             # single-field inflation PNG
                            'FFTLog mode':'FAST',
                            'k_pivot':0.05,
                            'P_k_max_h/Mpc':100.,
                            'tau_reio':0.0543,      # optical depth at reionization
                            'YHe':0.2454,           # Helium fraction?
                            'N_ur':2.0328,          # ?
                            'N_ncdm':1,             # 1 massive neutrino
                            'm_ncdm':0.06,          # mass of neutrino (eV)
                            'T_ncdm':0.71611,       # neutrino temperature?
                            'omega_b':self.ombh2_planck,
                            'n_s':self.ns_planck
                            }

#-------------------------------------------------------------------
# HELPER FUNCTIONS
#-------------------------------------------------------------------

    #-------------------------------------------------------------------
    # For generating individual elements of the Gaussian covariance matrix
    # see Survey_window_kernels.ipynb for further details where the same function is used
    def Cij(self, kt, Wij, Pfit):
        temp=np.zeros((7,6))
        for i in range(-3,4):
            if(kt+i<0 or kt+i>=self.kbins):
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
    def Dz(self, z,Om0):
        return(scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3)
                                    /scipy.special.hyp2f1(1/3., 1, 11/6., 1-1/Om0)/(1+z))

    #-------------------------------------------------------------------
    # Growth rate for LCDM cosmology
    def fgrowth(self, z,Om0):
        return(1. + 6*(Om0-1)*scipy.special.hyp2f1(4/3., 2, 17/6., (1-1/Om0)/(1+z)**3)
                    /( 11*Om0*(1+z)**3*scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3) ))

    #-------------------------------------------------------------------
    def Pk_lin(self, H0, omch2, ombh2, As, ns, z):
        """
        Generates a linear initial power spectrum from CAMB
        """
        #get matter power spectra and sigma8 at the redshift we want
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(ns=ns, As=np.exp(As)/1e10)
        #Note non-linear corrections couples to smaller scales than you want
        pars.set_matter_power(redshifts=[z], kmax=0.4)

        #Linear spectra
        pars.NonLinear = model.NonLinear_none
        results = camb.get_results(pars)
        # k bins will be interpolated to what we want later, so it's "ok" if this isn't exact
        # this k is in units of h/Mpc
        kh, z1, pk = results.get_matter_power_spectrum(minkh=0.0025, maxkh=0.4, npoints = 100)
        
        pdata = np.vstack((kh, pk[0])).T
        return pdata

    #-------------------------------------------------------------------
    def Pk_lin_CLASS(self, H0, omch2, ombh2, As, ns):
        """
        Generates a linear initial power spectrum with CLASS
        """
        cosmo = Class()
        cosmo.set(self.common_settings)

        cosmo.set({'A_s':np.exp(As)/1e10,
                'n_s':ns,
                'omega_b':ombh2,
                'omega_cdm':omch2,
                'H0':H0,
                'z_pk':self.z
                })  
        cosmo.compute()
        k = np.linspace(np.amin(self.k), np.amax(self.k), len(self.k)*3)
        khvec = k * cosmo.h()
        #get matter power spectra and sigma8 at the redshift we want
        pk_lin = np.asarray([cosmo.pk_lin(kh,self.z)*cosmo.h()**3. for kh in khvec])

        pdata = np.vstack((k, pk_lin)).T
        return pdata

    #-------------------------------------------------------------------
    def trispIntegrand(self, u12,k1,k2,Plin):
        return( (8*self.i44*(Plin(k1)**2*T0.e44o44_1(u12,k1,k2) + Plin(k2)**2*T0.e44o44_1(u12,k2,k1))
                +16*self.i44*Plin(k1)*Plin(k2)*T0.e44o44_2(u12,k1,k2)
                +8*self.i34*(Plin(k1)*T0.e34o44_2(u12,k1,k2)+Plin(k2)*T0.e34o44_2(u12,k2,k1))
                +2*self.i24*T0.e24o44(u12,k1,k2))
                *Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12)) )

    #-------------------------------------------------------------------
    # Returns the tree-level trispectrum as a function of multipoles and k1, k2
    def trisp(self, l1,l2,k1,k2, Plin):
        T0.l1=l1; T0.l2=l2
        expr = self.i44*(Plin(k1)**2*Plin(k2)*T0.ez3(k1,k2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2,k1))\
            +8*self.i34*Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2)

        temp = (quad(self.trispIntegrand, -1, 1,args=(k1,k2,Plin), limit=60)[0]/2. + expr)/self.i22**2
        return(temp)

    #-------------------------------------------------------------------
    # Using the Z12 kernel which is defined in Eq. (A9) (equations copy-pasted from Generating_T0_Z12_expressions.nb)
    def Z12Kernel(self, l,mu,be,b1,b2,g2,dlnpk):
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
    def lp(self, l,mu):
        if (l==0): exp=1
        if (l==2): exp=((3*mu**2 - 1)/2.)
        if (l==4): exp=((35*mu**4 - 30*mu**2 + 3)/8.)
        return(exp)

    #-------------------------------------------------------------------
    # For transforming the linear array to a matrix
    def MatrixForm(self, a):
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
    def Z12Multipoles(self, i,l,be,b1,b2,g2,dlnpk):
        return(quad(lambda mu: self.lp(i,mu)*self.Z12Kernel(l,mu,be,b1,b2,g2,dlnpk), -1, 1, limit=60)[0])

    #-------------------------------------------------------------------
    def CovLATerm(self, sigma22x10, dlnPk, be,b1,b2,g2):
        """
        Calculates the LA terms used in the SSC calculations
        """
        covaLAterm=np.zeros((3,len(self.k)))
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    covaLAterm[l]+=1/4.*sigma22x10[i,j]*self.vec_Z12Multipoles(2*i,2*l,be,b1,b2,g2,dlnPk)\
                    *quad(lambda mu: self.lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]
        return covaLAterm
            
    #-------------------------------------------------------------------
    def covaSSC(self, l1,l2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk):
        """
        Returns the SSC covariance matrix contrbution
        BC= beat-coupling, LA= local average effect, SSC= super sample covariance
        """
        
        covaBC=np.zeros((len(self.k),len(self.k)))
        for i in range(3):
            for j in range(3):
                covaBC+=1/4.*sigma22Sq[i,j]*np.outer(Plin(self.k)*self.vec_Z12Multipoles(2*i,l1,be,b1,b2,g2,dlnPk),Plin(self.k)*self.vec_Z12Multipoles(2*j,l2,be,b1,b2,g2,dlnPk))
                sigma10Sq[i,j]=1/4.*sigma10Sq[i,j]*quad(lambda mu: self.lp(2*i,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]\
                *quad(lambda mu: self.lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]

        covaLA=-rsd[l2]*np.outer(Plin(self.k)*(covaLAterm[int(l1/2)]+self.i32/self.i22/self.i10*rsd[l1]*Plin(self.k)*b2/b1**2+2/self.i10*rsd[l1]),Plin(self.k))\
            -rsd[l1]*np.outer(Plin(self.k),Plin(self.k)*(covaLAterm[int(l2/2)]+self.i32/self.i22/self.i10*rsd[l2]*Plin(self.k)*b2/b1**2+2/self.i10*rsd[l2]))\
            +(np.sum(sigma10Sq)+1/self.i10)*rsd[l1]*rsd[l2]*np.outer(Plin(self.k),Plin(self.k))

        return(covaBC+covaLA)

    #-------------------------------------------------------------------
    # Functions meant to be called elsewhere
    # ------------------------------------------------------------------

    #-------------------------------------------------------------------
    def Pk_CLASS_PT(self, params):

        cosmo = Class()
        # unpack / specify parameters
        H0    = params[0]
        omch2 = params[1]
        As    = params[2] * self.A_planck

        b1    = params[3]
        b2    = params[4]
        bG2   = params[5]
        bGamma3 = 0 # set to 0 since multipoles can only distinguish bG2 + 0.4*bGamma3
        # we're only using monopole+quadropole, so the specific value for this "shouldn't" matter
        cs4 = 0.
        # NOTE: I'm pretty sure b4 is actually cbar
        # these parameters are normally analytically marginalized over, but you can specify them if don't want to do that
        cs0, cs2, b4, Pshot  = params[6], params[7], params[8], params[9]
        a0 = 0
        a2 = 0

        # set cosmology parameters
        cosmo.set(self.common_settings)
        cosmo.set({'A_s':np.exp(As)/1e10,
                'omega_cdm':omch2,
                'H0':H0,
                'z_pk':self.z
                })  
        try: cosmo.compute()
        except: return []

        # theory calculations taken from Misha Ivanov's likelihood function
        norm = 1
        h = cosmo.h()
        fz = cosmo.scale_independent_growth_factor_f(self.z)
        all_theory = cosmo.get_pk_mult(self.kbins3*h,self.z, 400)
        kinloop1 = self.kbins3 * h

        theory4 = ((norm**2*all_theory[20] + norm**4*all_theory[27]
                + b1*norm**3*all_theory[28]
                + b1**2*norm**2*all_theory[29]
                + b2*norm**3*all_theory[38]
                + bG2*norm**3*all_theory[39]
                + 2.*cs4*norm**2*all_theory[13]/h**2)*h**3
                + fz**2*b4*self.kbins3**2*(norm**2*fz**2*48./143. + 48.*fz*b1*norm/77.+8.*b1**2/35.)*(35./8.)*all_theory[13]*h)

        theory2 = ((norm**2*all_theory[18]
                + norm**4*(all_theory[24])
                + norm**1*b1*all_theory[19]
                + norm**3*b1*(all_theory[25])
                + b1**2*norm**2*all_theory[26]
                + b1*b2*norm**2*all_theory[34]
                + b2*norm**3*all_theory[35]
                + b1*bG2*norm**2*all_theory[36]
                + bG2*norm**3*all_theory[37]
                + 2.*(cs2 + 0.*b4*kinloop1**2)*norm**2*all_theory[12]/h**2
                + (2.*bG2+0.8*bGamma3*norm)*norm**3*all_theory[9])*h**3
                + a2*(2./3.)*(self.kbins3/0.45)**2.
                + fz**2*b4*self.kbins3**2*((norm**2*fz**2*70. + 165.*fz*b1*norm+99.*b1**2)*4./693.)*(35./8.)*all_theory[13]*h)

        theory0 = ((norm**2*all_theory[15]
                + norm**4*(all_theory[21])
                + norm**1*b1*all_theory[16]
                + norm**3*b1*(all_theory[22])
                + norm**0*b1**2*all_theory[17]
                + norm**2*b1**2*all_theory[23]
                + 0.25*norm**2*b2**2*all_theory[1]
                + b1*b2*norm**2*all_theory[30]
                + b2*norm**3*all_theory[31]
                + b1*bG2*norm**2*all_theory[32]
                + bG2*norm**3*all_theory[33]
                + b2*bG2*norm**2*all_theory[4]
                + bG2**2*norm**2*all_theory[5]
                + 2.*cs0*norm**2*all_theory[11]/h**2
                + (2.*bG2+0.8*bGamma3*norm)*norm**2*(b1*all_theory[7]+norm*all_theory[8]))*h**3
                + Pshot
                + a0*(self.kbins3/0.45)**2.
                + a2*(1./3.)*(self.kbins3/0.45)**2.
                + fz**2*b4*self.kbins3**2*(norm**2*fz**2/9. + 2.*fz*b1*norm/7. + b1**2/5)*(35./8.)*all_theory[13]*h)

        pk_g0 = InterpolatedUnivariateSpline(self.kbins3,theory0)(self.k)
        pk_g2 = InterpolatedUnivariateSpline(self.kbins3,theory2)(self.k)
        pk_g4 = InterpolatedUnivariateSpline(self.kbins3,theory4)(self.k)

        # This line is necesary to prevent memory leaks
        cosmo.struct_cleanup()

        return [pk_g0, 0, pk_g2, 0, pk_g4]

#-------------------------------------------------------------------
    def Pk_CLASS_PT_2(self, params, k, return_sigma8=False):
        """
        Same function as above, except with user-specified k-bins
        """
        cosmo = Class()
        # unpack / specify parameters
        H0    = params[0]
        omch2 = params[1]
        ombh2 = self.ombh2_planck
        As    = params[2] * self.A_planck
        ns    = self.ns_planck

        b1    = params[3]
        b2    = params[4]
        bG2   = params[5]
        bGamma3 = 0 # set to 0 since multipoles can only distinguish bG2 + 0.4*bGamma3
        # we're only using monopole+quadropole, so the specific value for this "shouldn't" matter
        cs4 = -5.
        # NOTE: I'm pretty sure b4 is actually cbar
        cs0   = 0.
        cs2   = 0.
        cbar  = 500.
        Pshot = 0

        # set cosmology parameters
        cosmo.set(self.common_settings)
        cosmo.set({'A_s':np.exp(As)/1e10,
                'n_s':ns,
                'omega_b':ombh2,
                'omega_cdm':omch2,
                'H0':H0,
                'z_pk':self.z
                })  
        try: cosmo.compute()
        except: return []

        cosmo.initialize_output(k*cosmo.h(), self.z, len(k))

        #print(b1, b2, bG2, bGamma3, cs0, Pshot, cbar)
        # calculate galaxy power spectrum multipoles
        pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, cbar)
        pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, cbar)
        pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, cbar)

        sigma8 = cosmo.sigma8()
        # This line is necesary to prevent memory leaks
        cosmo.struct_cleanup()

        if return_sigma8 == False: return np.concatenate([pk_g0, pk_g2, pk_g4])
        else: return np.concatenate([pk_g0, pk_g2, pk_g4]), sigma8

    #-------------------------------------------------------------------
    def get_k_bins(self):
        return self.k

    #-------------------------------------------------------------------
    def get_gaussian_covariance(self, params, return_Pk=False, Pk_galaxy=[]):
        """
        Returns the (Monopole+Quadrupole) Gaussian covariance matrix
        If Pk_galaxy is already calculated, takes ~10 ms to run
        Else, takes ~0.5 s
        """
        # generate galaxy redshift-space power spectrum if necesary
        if len(Pk_galaxy) == 0:
            Pk_galaxy = self.Pk_CLASS_PT(params)
        # if generating the galaxy power spectrum failed, return nan value
        if len(Pk_galaxy) == 0:
            if return_Pk == False: return np.nan
            else: return np.nan, np.nan

        # sanity check to make sure the galaxy power spectrum has the correct dimensions
        assert len(Pk_galaxy[0]) == len(self.k), "Galaxy power spectrum has wrong dimensions! Double check your k-bins"

        covMat=np.zeros((2*self.kbins,2*self.kbins))
        for i in range(self.kbins):
            temp=self.Cij(i,self.WijFile[i], Pk_galaxy)
            C00=temp[:,0]; C22=temp[:,1]; C20=temp[:,3]
            for j in range(-3,4):
                if(i+j>=0 and i+j<self.kbins):
                    covMat[i,i+j]=C00[j+3]
                    covMat[self.kbins+i,self.kbins+i+j]=C22[j+3]
                    covMat[self.kbins+i,i+j]=C20[j+3]
        covMat[:self.kbins,self.kbins:self.kbins*2]=np.transpose(covMat[self.kbins:self.kbins*2,:self.kbins])
        covMat=(covMat+np.transpose(covMat))/2.

        if return_Pk == False: return covMat
        else: return covMat, Pk_galaxy

    #-------------------------------------------------------------------
    def get_non_gaussian_covariance(self, params, do_T0=True):
        """
        Returns the Non-Gaussian portion of the covariance matrix
        Takes ~ 10 minutes to run
        """
        # unpack parameters
        H0, omch2, A, b1, b2 = params[0], params[1], params[2], params[3], params[4]
        ombh2 = self.ombh2_planck
        ns = self.ns_planck
        As = A * self.A_planck
        b1 = b1
        b2 = b2

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
        be = self.fgrowth(self.z, Omega_m)/b1; #beta = f/b1, zero for real space

        # initializing bias parameters for trispectrum
        T0.InitParameters([b1,be,g2,b2,g3,g2x,g21,b3])

        # Get initial power spectrum
        pdata = self.Pk_lin_CLASS(H0, omch2, ombh2, As, ns)
        Plin=InterpolatedUnivariateSpline(pdata[:,0], self.Dz(self.z, Omega_m)**2*b1**2*pdata[:,1])

        # Get the derivative of the linear power spectrum
        dlnPk=derivative(Plin,self.k,dx=1e-4)*self.k/Plin(self.k)
        
        # Kaiser terms
        rsd=np.zeros(5)
        rsd[0]=1 + (2*be)/3 + be**2/5
        rsd[2]=(4*be)/3 + (4*be**2)/7
        rsd[4]=(8*be**2)/35
        
        # Calculating the RMS fluctuations of supersurvey modes 
        #(e.g., sigma22Sq which was defined in Eq. (33) and later calculated in Eq.(65)
        kwin = self.powW22[:,0]
        [temp,temp2]=np.zeros((2,6)); temp3 = np.zeros(9)
        for i in range(9):
            Pwin=InterpolatedUnivariateSpline(kwin, self.powW22x10[:,1+i])
            temp3[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]

            if(i<6):
                Pwin=InterpolatedUnivariateSpline(kwin, self.powW22[:,1+i])
                temp[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]
                Pwin=InterpolatedUnivariateSpline(kwin, self.powW10[:,1+i])
                temp2[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, kwin[-1], limit=100)[0]
            else:
                continue
        
        sigma22Sq = self.MatrixForm(temp); sigma10Sq = self.MatrixForm(temp2); sigma22x10 = self.MatrixForm(temp3)
    
        # Calculate the LA term
        covaLAterm = self.CovLATerm(sigma22x10, dlnPk, be,b1,b2,g2)
        
        covaSSCmult=np.zeros((2*self.kbins,2*self.kbins))
        covaSSCmult[:self.kbins,:self.kbins]=self.covaSSC(0,0, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
        covaSSCmult[self.kbins:,self.kbins:]=self.covaSSC(2,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
        covaSSCmult[:self.kbins,self.kbins:]=self.covaSSC(0,2, covaLAterm, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk); 
        covaSSCmult[self.kbins:,:self.kbins]=np.transpose(covaSSCmult[:self.kbins,self.kbins:])

        if do_T0 == False:
            return covaSSCmult, None

        # Calculate the Non-Gaussian multipole covariance
        # Warning: the trispectrum takes a while to run
        covaT0mult=np.zeros((2*self.kbins,2*self.kbins))
        for i in range(len(self.k)):
            covaT0mult[i,:self.kbins]=self.vec_trisp(0,0,self.k[i],self.k, Plin)
            covaT0mult[i,self.kbins:]=self.vec_trisp(0,2,self.k[i],self.k, Plin)
            covaT0mult[self.kbins+i,self.kbins:]=self.vec_trisp(2,2,self.k[i],self.k, Plin)

        covaT0mult[self.kbins:,:self.kbins]=np.transpose(covaT0mult[:self.kbins,self.kbins:])

        #return covaNG
        return covaSSCmult, covaT0mult

    #-------------------------------------------------------------------
    def get_full_covariance(self, params, Pk_galaxy=[]):
        """
        Returns the full analytic covariance matrix
        """
        cov_G = self.get_gaussian_covariance(params, False, Pk_galaxy)
        cov_SSC, cov_T0 = self.get_non_gaussian_covariance(params)
        return cov_G, cov_SSC, cov_T0
