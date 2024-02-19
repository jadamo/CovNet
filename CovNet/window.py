# This file is a repackaging of the "Survey_window_kernels.ipynb" found in CovaPT
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import os
import numpy as np
from numpy import conj

from CovNet.config import CovaPT_data_dir

class Gaussian_Window_Kernels():

    def __init__(self, k_centers, key="HighZ_NGC"):
        """
        Constructs window kernel object 
        @param {np array} array of k bin centers
        """

        assert key in ["HighZ_NGC", "HighZ_SGC", "LowZ_NGZ", "LowZ_SGC"], \
               'ERROR: invalid key specified! Should be one of ["HighZ_NGC", "HighZ_SGC", "LowZ_NGZ", "LowZ_SGC"]'

        # the total number of kbins
        self.nBins = len(k_centers)

        # calculate bin edges and width from the k centers
        self.kbin_width, self.kbin_edges = self.get_k_bin_edges(k_centers)

        # length of box when computing FFTs
        self.Lbox = 3750.

        # fundamental k mode
        self.kfun=2.*np.pi/self.Lbox

        # As the window falls steeply with k, only low-k regions are needed for the calculation.
        # Therefore cutting out the high-k modes in the FFTs using the self.icut parameter
        self.icut=15; # needs to be less than Lm//2 (Lm: size of FFT)

        # from eq 3 of ??
        self.I22=437.183365

        self.Lm2 = int(self.kbin_width*self.nBins/self.kfun)+1
        assert self.icut < self.Lm2

        # Load survey random FFTs
        # TODO: Update this with code to calculate FFTs
        self.fft_file = CovaPT_data_dir+'FFTWinFun_HighZ_NGC.npy'
        self.Wij2 = self.load_fft_file()

    def get_k_bin_edges(self, k_centers):
        kbin_width = k_centers[-1] - k_centers[-2]
        kbin_half_width = kbin_width / 2.
        kbin_edges = np.zeros(len(k_centers)+1)
        kbin_edges[0] = k_centers[0] - kbin_half_width

        assert kbin_edges[0] > 0.
        for i in range(1, len(kbin_edges)):
            kbin_edges[i] = k_centers[i-1] + kbin_half_width

        return kbin_width, kbin_edges

    def fft(self, temp):
        """
        Does some shifting of the fft arrays
        """
        ia=self.Lm//2-1; ib=self.Lm//2+1
        temp2=np.zeros((self.Lm,self.Lm,self.Lm),dtype='<c8')
        temp2[ia:self.Lm,ia:self.Lm,ia:self.Lm]=temp[0:ib,0:ib,0:ib]; temp2[0:ia,ia:self.Lm,ia:self.Lm]=temp[ib:self.Lm,0:ib,0:ib]
        temp2[ia:self.Lm,0:ia,ia:self.Lm]=temp[0:ib,ib:self.Lm,0:ib]; temp2[ia:self.Lm,ia:self.Lm,0:ia]=temp[0:ib,0:ib,ib:self.Lm]
        temp2[0:ia,0:ia,ia:self.Lm]=temp[ib:self.Lm,ib:self.Lm,0:ib]; temp2[0:ia,ia:self.Lm,0:ia]=temp[ib:self.Lm,0:ib,ib:self.Lm]
        temp2[ia:self.Lm,0:ia,0:ia]=temp[0:ib,ib:self.Lm,ib:self.Lm]; temp2[0:ia,0:ia,0:ia]=temp[ib:self.Lm,ib:self.Lm,ib:self.Lm]
    
        return(temp2[ia-self.icut:ia+self.icut+1,ia-self.icut:ia+self.icut+1,ia-self.icut:ia+self.icut+1])

    def get_shell_modes(self):
        [ix,iy,iz] = np.zeros((3,2*self.Lm2+1,2*self.Lm2+1,2*self.Lm2+1));
        Bin_kmodes=[]; Bin_ModeNum=np.zeros(self.nBins,dtype=int)

        for i in range(self.nBins): Bin_kmodes.append([])
        for i in range(len(ix)):
            ix[i,:,:]+=i-self.Lm2; iy[:,i,:]+=i-self.Lm2; iz[:,:,i]+=i-self.Lm2

        rk=np.sqrt(ix**2+iy**2+iz**2)
        sort=(rk*self.kfun/self.kbin_width).astype(int)

        for i in range(self.nBins):
            ind=(sort==i); Bin_ModeNum[i]=len(ix[ind]); \
            Bin_kmodes[i]=np.hstack((ix[ind].reshape(-1,1),iy[ind].reshape(-1,1),iz[ind].reshape(-1,1),rk[ind].reshape(-1,1)))
        return Bin_kmodes, Bin_ModeNum

    def load_fft_file(self):
        """
        Loads and organizes information from the random catalog FFTs
        """
        Wij = np.load(self.fft_file)
        self.Lm=len(Wij[0]) #size of FFT

        Wij2=[]
        for i in range(len(Wij)//2): #W22
            Wij2.append(self.fft(Wij[i]))

        for i in range(len(Wij)//2,len(Wij)): #W12, I'm taking conjugate as that is used in the 'WinFun' function later
            Wij2.append(conj(self.fft(Wij[i])))

        return Wij2
    
    def calc_gaussian_window_function(self, bin_idx, kmodes_sampled=400):
        """
        Returns the window function kernels for l=0,2,and 4 auto + cross covariance
        NOTE: This function is computationally expensive and should be run on a cluster in parralel
        """

        Bin_kmodes, Bin_ModeNum = self.get_shell_modes()

        [W, Wxx, Wxy, Wxz, Wyy, Wyz, Wzz, Wxxxx, Wxxxy, Wxxxz, Wxxyy, Wxxyz, Wxxzz, Wxyyy, Wxyyz, Wxyzz,\
        Wxzzz, Wyyyy, Wyyyz, Wyyzz, Wyzzz, Wzzzz, W12, W12xx, W12xy, W12xz, W12yy, W12yz, W12zz, W12xxxx,\
        W12xxxy, W12xxxz, W12xxyy, W12xxyz, W12xxzz, W12xyyy, W12xyyz, W12xyzz, W12xzzz, W12yyyy, W12yyyz,\
        W12yyzz, W12yzzz, W12zzzz] = self.Wij2

        avgWij=np.zeros((2*3+1,15,6)); avgW00=np.zeros((2*3+1,15),dtype='<c8');
        avgW22=avgW00.copy(); avgW44=avgW00.copy(); avgW20=avgW00.copy(); avgW40=avgW00.copy(); avgW42=avgW00.copy()
        [ix,iy,iz,k2xh,k2yh,k2zh]=np.zeros((6,2*self.icut+1,2*self.icut+1,2*self.icut+1))
        
        for i in range(2*self.icut+1): 
            ix[i,:,:]+=i-self.icut; iy[:,i,:]+=i-self.icut; iz[:,:,i]+=i-self.icut
            
        if (kmodes_sampled<Bin_ModeNum[bin_idx]):
            norm=kmodes_sampled
            sampled=(np.random.rand(kmodes_sampled)*Bin_ModeNum[bin_idx]).astype(int)
        else:
            norm=Bin_ModeNum[bin_idx]
            sampled=np.arange(Bin_ModeNum[bin_idx],dtype=int)
        
        # Randomly select a mode in the k1 bin
        for n in sampled:
            [ik1x,ik1y,ik1z,rk1]=Bin_kmodes[bin_idx][n]
            if (rk1==0.): k1xh=0; k1yh=0; k1zh=0
            else: k1xh=ik1x/rk1; k1yh=ik1y/rk1; k1zh=ik1z/rk1
                
        # Build a 3D array of modes around the selected mode   
            k2xh=ik1x-ix; k2yh=ik1y-iy; k2zh=ik1z-iz
            rk2=np.sqrt(k2xh**2+k2yh**2+k2zh**2)
            sort=(rk2*self.kfun/self.kbin_width).astype(int)-bin_idx # to decide later which shell the k2 mode belongs to
            ind=(rk2==0)
            if (ind.any()>0): rk2[ind]=1e10
            k2xh/=rk2; k2yh/=rk2; k2zh/=rk2;
            #k2 hat arrays built
            
        # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
        # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
        # L(i) refers to multipoles
            
            W_L0 = W
            Wc_L0 = conj(W)
            
            xx=Wxx*k1xh**2+Wyy*k1yh**2+Wzz*k1zh**2+2.*Wxy*k1xh*k1yh+2.*Wyz*k1yh*k1zh+2.*Wxz*k1zh*k1xh
            
            W_k1L2=1.5*xx-0.5*W
            W_k2L2=1.5*(Wxx*k2xh**2+Wyy*k2yh**2+Wzz*k2zh**2 \
            +2.*Wxy*k2xh*k2yh+2.*Wyz*k2yh*k2zh+2.*Wxz*k2zh*k2xh)-0.5*W
            Wc_k1L2=conj(W_k1L2)
            Wc_k2L2=conj(W_k2L2)
            
            W_k1L4=35./8.*(Wxxxx*k1xh**4 +Wyyyy*k1yh**4+Wzzzz*k1zh**4 \
        +4.*Wxxxy*k1xh**3*k1yh +4.*Wxxxz*k1xh**3*k1zh +4.*Wxyyy*k1yh**3*k1xh \
        +4.*Wyyyz*k1yh**3*k1zh +4.*Wxzzz*k1zh**3*k1xh +4.*Wyzzz*k1zh**3*k1yh \
        +6.*Wxxyy*k1xh**2*k1yh**2+6.*Wxxzz*k1xh**2*k1zh**2+6.*Wyyzz*k1yh**2*k1zh**2 \
        +12.*Wxxyz*k1xh**2*k1yh*k1zh+12.*Wxyyz*k1yh**2*k1xh*k1zh +12.*Wxyzz*k1zh**2*k1xh*k1yh) \
        -5./2.*W_k1L2 -7./8.*W_L0
            Wc_k1L4=conj(W_k1L4)
            
            k1k2=Wxxxx*(k1xh*k2xh)**2+Wyyyy*(k1yh*k2yh)**2+Wzzzz*(k1zh*k2zh)**2 \
                +Wxxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +Wxxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +Wyyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +Wyzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +Wxyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +Wxzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +Wxxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +Wxxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +Wyyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +Wxyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +Wxxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +Wxyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            W_k2L4=35./8.*(Wxxxx*k2xh**4 +Wyyyy*k2yh**4+Wzzzz*k2zh**4 \
        +4.*Wxxxy*k2xh**3*k2yh +4.*Wxxxz*k2xh**3*k2zh +4.*Wxyyy*k2yh**3*k2xh \
        +4.*Wyyyz*k2yh**3*k2zh +4.*Wxzzz*k2zh**3*k2xh +4.*Wyzzz*k2zh**3*k2yh \
        +6.*Wxxyy*k2xh**2*k2yh**2+6.*Wxxzz*k2xh**2*k2zh**2+6.*Wyyzz*k2yh**2*k2zh**2 \
        +12.*Wxxyz*k2xh**2*k2yh*k2zh+12.*Wxyyz*k2yh**2*k2xh*k2zh +12.*Wxyzz*k2zh**2*k2xh*k2yh) \
        -5./2.*W_k2L2 -7./8.*W_L0
            Wc_k2L4=conj(W_k2L4)
            
            W_k1L2_k2L2= 9./4.*k1k2 -3./4.*xx -1./2.*W_k2L2
            W_k1L2_k2L4=2/7.*W_k1L2+20/77.*W_k1L4 #approximate as 6th order FFTs not simulated
            W_k1L4_k2L2=W_k1L2_k2L4 #approximate
            W_k1L4_k2L4=1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4
            Wc_k1L2_k2L2= conj(W_k1L2_k2L2)
            Wc_k1L2_k2L4=conj(W_k1L2_k2L4); Wc_k1L4_k2L2=Wc_k1L2_k2L4
            Wc_k1L4_k2L4=conj(W_k1L4_k2L4)
            
            k1k2W12=W12xxxx*(k1xh*k2xh)**2+W12yyyy*(k1yh*k2yh)**2+W12zzzz*(k1zh*k2zh)**2 \
                +W12xxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +W12xxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +W12yyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +W12yzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +W12xyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +W12xzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +W12xxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +W12xxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +W12yyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +W12xyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +W12xxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +W12xyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            xxW12=W12xx*k1xh**2+W12yy*k1yh**2+W12zz*k1zh**2+2.*W12xy*k1xh*k1yh+2.*W12yz*k1yh*k1zh+2.*W12xz*k1zh*k1xh
        
            W12_L0 = W12
            W12_k1L2=1.5*xxW12-0.5*W12
            W12_k1L4=35./8.*(W12xxxx*k1xh**4 +W12yyyy*k1yh**4+W12zzzz*k1zh**4 \
        +4.*W12xxxy*k1xh**3*k1yh +4.*W12xxxz*k1xh**3*k1zh +4.*W12xyyy*k1yh**3*k1xh \
        +6.*W12xxyy*k1xh**2*k1yh**2+6.*W12xxzz*k1xh**2*k1zh**2+6.*W12yyzz*k1yh**2*k1zh**2 \
        +12.*W12xxyz*k1xh**2*k1yh*k1zh+12.*W12xyyz*k1yh**2*k1xh*k1zh +12.*W12xyzz*k1zh**2*k1xh*k1yh) \
        -5./2.*W12_k1L2 -7./8.*W12_L0
            W12_k1L4_k2L2=2/7.*W12_k1L2+20/77.*W12_k1L4
            W12_k1L4_k2L4=1/9.*W12_L0+100/693.*W12_k1L2+162/1001.*W12_k1L4
            W12_k2L2=1.5*(W12xx*k2xh**2+W12yy*k2yh**2+W12zz*k2zh**2\
            +2.*W12xy*k2xh*k2yh+2.*W12yz*k2yh*k2zh+2.*W12xz*k2zh*k2xh)-0.5*W12
            W12_k2L4=35./8.*(W12xxxx*k2xh**4 +W12yyyy*k2yh**4+W12zzzz*k2zh**4 \
        +4.*W12xxxy*k2xh**3*k2yh +4.*W12xxxz*k2xh**3*k2zh +4.*W12xyyy*k2yh**3*k2xh \
        +4.*W12yyyz*k2yh**3*k2zh +4.*W12xzzz*k2zh**3*k2xh +4.*W12yzzz*k2zh**3*k2yh \
        +6.*W12xxyy*k2xh**2*k2yh**2+6.*W12xxzz*k2xh**2*k2zh**2+6.*W12yyzz*k2yh**2*k2zh**2 \
        +12.*W12xxyz*k2xh**2*k2yh*k2zh+12.*W12xyyz*k2yh**2*k2xh*k2zh +12.*W12xyzz*k2zh**2*k2xh*k2yh) \
        -5./2.*W12_k2L2 -7./8.*W12_L0
            
            W12_k1L2_k2L2= 9./4.*k1k2W12 -3./4.*xxW12 -1./2.*W12_k2L2
            
            W_k1L2_Sumk2L22=1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24=2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22=1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24=2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44=1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4
            
            C00exp = [Wc_L0*W_L0,Wc_L0*W_k2L2,Wc_L0*W_k2L4,\
                    Wc_k1L2*W_L0,Wc_k1L2*W_k2L2,Wc_k1L2*W_k2L4,\
                    Wc_k1L4*W_L0,Wc_k1L4*W_k2L2,Wc_k1L4*W_k2L4]
            
            C00exp += [2.*W_L0*W12_L0,W_k1L2*W12_L0,W_k1L4*W12_L0,\
                    W_k2L2*W12_L0,W_k2L4*W12_L0,conj(W12_L0)*W12_L0]
            
            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,\
                    Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,\
                    Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,\
                    Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,\
                    Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]
            
            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2\
                    +W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,\
                    0.5*((1/5.*W_L0+2/7.*W_k1L2+18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2\
    +(1/5.*W_k2L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),\
        0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2\
    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),\
    0.5*(W_k1L2_k2L2*W12_k2L2+(1/5.*W_L0+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L2\
    +(1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),\
    0.5*(W_k1L2_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L2\
    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4)*W12_L0 + W_k2L4*W12_k1L2_k2L2),\
                    conj(W12_k1L2_k2L2)*W12_L0+conj(W12_k1L2)*W12_k2L2]
            
            C44exp = [Wc_k2L4*W_k1L4 + Wc_L0*W_k1L4_k2L4,\
                    Wc_k2L4*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k2L4*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L44,\
                    Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,\
                    Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,\
                    Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]
            
            C44exp += [W_k1L4*W12_k2L4 + W_k2L4*W12_k1L4\
                    +W_k1L4_k2L4*W12_L0+W_L0*W12_k1L4_k2L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4\
    +(2/7.*W_k1L2_k2L4+20/77.*W_k1L4_k2L4)*W12_L0 + W_k1L2*W12_k1L4_k2L4),\
    0.5*((1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4\
    +(1/9.*W_k2L4+100/693.*W_k1L2_k2L4+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k1L4*W12_k1L4_k2L4),\
    0.5*(W_k1L4_k2L2*W12_k2L4+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L4),\
    0.5*(W_k1L4_k2L4*W12_k2L4+(1/9.*W_L0+100/693.*W_k2L2+162/1001.*W_k2L4)*W12_k1L4\
    +(1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L4),\
                    conj(W12_k1L4_k2L4)*W12_L0+conj(W12_k1L4)*W12_k2L4] #1/(nbar)^2
            
            C20exp = [Wc_L0*W_k1L2,Wc_L0*W_k1L2_k2L2,Wc_L0*W_k1L2_k2L4,\
                    Wc_k1L2*W_k1L2,Wc_k1L2*W_k1L2_k2L2,Wc_k1L2*W_k1L2_k2L4,\
                    Wc_k1L4*W_k1L2,Wc_k1L4*W_k1L2_k2L2,Wc_k1L4*W_k1L2_k2L4]
            
            C20exp += [W_k1L2*W12_L0 + W*W12_k1L2,\
                    0.5*((1/5.*W+2/7.*W_k1L2+18/35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L2),\
                    0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),\
                    0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),\
                    np.conj(W12_k1L2)*W12_L0]
            
            C40exp = [Wc_L0*W_k1L4,Wc_L0*W_k1L4_k2L2,Wc_L0*W_k1L4_k2L4,\
                    Wc_k1L2*W_k1L4,Wc_k1L2*W_k1L4_k2L2,Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L4*W_k1L4,Wc_k1L4*W_k1L4_k2L2,Wc_k1L4*W_k1L4_k2L4]
            
            C40exp += [W_k1L4*W12_L0 + W*W12_k1L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L4),\
                    0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),\
                    0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),\
                    0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),\
                    np.conj(W12_k1L4)*W12_L0]
            
            C42exp = [Wc_k2L2*W_k1L4 + Wc_L0*W_k1L4_k2L2,\
                    Wc_k2L2*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L22,\
                    Wc_k2L2*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,\
                    Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,\
                    Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]
            
            C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4+\
                    W_k1L4_k2L2*W12_L0+W*W12_k1L4_k2L2,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4\
        +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L4_k2L2),\
        0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4\
    +(1/9.*W_k2L2+100/693.*W_k1L2_k2L2+162/1001.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L4_k2L2),\
    0.5*(W_k1L4_k2L2*W12_k2L2+(1/5.*W+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L4\
    +(1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L2),\
    0.5*(W_k1L4_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L2),\
                    conj(W12_k1L4_k2L2)*W12_L0+conj(W12_k1L4)*W12_k2L2] #1/(nbar)^2
            
            for i in range(-3,4):
                ind=(sort==i)
                for j in range(15):
                    avgW00[i+3,j]+=np.sum(C00exp[j][ind])
                    avgW22[i+3,j]+=np.sum(C22exp[j][ind])
                    avgW44[i+3,j]+=np.sum(C44exp[j][ind])
                    avgW20[i+3,j]+=np.sum(C20exp[j][ind])
                    avgW40[i+3,j]+=np.sum(C40exp[j][ind])
                    avgW42[i+3,j]+=np.sum(C42exp[j][ind])
                
        for i in range(0,2*3+1):
            if(i+bin_idx-3>=self.nBins or i+bin_idx-3<0): 
                avgW00[i]*=0; avgW22[i]*=0; avgW44[i]*=0;
                avgW20[i]*=0; avgW40[i]*=0; avgW42[i]*=0; continue
            avgW00[i]=avgW00[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            avgW22[i]=avgW22[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            avgW44[i]=avgW44[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            avgW20[i]=avgW20[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            avgW40[i]=avgW40[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            avgW42[i]=avgW42[i]/(norm*Bin_ModeNum[bin_idx+i-3]*self.I22**2)
            
        avgWij[:,:,0]=2.*np.real(avgW00); avgWij[:,:,1]=25.*np.real(avgW22); avgWij[:,:,2]=81.*np.real(avgW44);
        avgWij[:,:,3]=5.*2.*np.real(avgW20); avgWij[:,:,4]=9.*2.*np.real(avgW40); avgWij[:,:,5]=45.*np.real(avgW42);
        
        return(avgWij)