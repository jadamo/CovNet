import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
from tqdm import tqdm
from classy import Class
import pymc3 as pm
import theano
import theano.tensor as tt
from contextlib import redirect_stdout

sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet, CovaPT
import tracemalloc

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set-NG/"
data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"

use_T0 = True

vary_covariance = False

C_fid_file = np.load(data_dir+"Cov_Fid.npz")
if use_T0 == True:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"] + C_fid_file["C_T0"]
else:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"]

P_fid = np.linalg.inv(C_fid)

cosmo_prior = np.array([[66.0, 75.5],
                        [0.10782, 0.13178],
                        [0.0211375, 0.0233625],
                        [2.4752, 3.7128],#[1.1885e-9, 2.031e-9],
                        [1.9, 2.45],
                        [-3.562, 0.551]])

#                     H0,omch2, ombh2,  As,  b1,     b2
cosmo_fid = np.array([67.8,0.1190,0.02215,3.094,1.9485,-0.5387])

gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}
redshift = 0.58

def PCA_emulator():
    """
    Sets up the PCA covariance matrix emulator to use in likelihood analysis
    """
    N_C = 100
    C_PCA = np.zeros((N_C, 100, 100))
    params_PCA = np.zeros((N_C, 6))
    for i in range(N_C):
        temp = np.load(PCA_dir+"CovNG-"+f'{i:04d}'+".npz")
        params_PCA[i] = temp["params"]
        C_PCA[i] = temp["C"]
    
        # if the matrix doesn't match the transpose close enough, manually flip over the diagonal
        try:
            np.testing.assert_allclose(C_PCA[i], C_PCA[i].T, err_msg="covariance must match transpose")
        except AssertionError:
            L = np.tril(C_PCA[i])
            U = np.tril(C_PCA[i], k=-1).T
            C_PCA[i] = L + U
    try:
        Emu = ce.CovEmu(params_PCA, C_PCA, NPC_D=20, NPC_L=20)
    except np.linalg.LinAlgError:
        print("WARNING! PCA emulator unable to initialize - input matrices aren't positive definite!")
        Emu = None
    return Emu

def CovNet_emulator():
    """
    Returns a neural net covariance matrix emulator
    """
    VAE = CovNet.Network_VAE(True).to(CovNet.try_gpu());       VAE.eval()
    decoder = CovNet.Block_Decoder(True).to(CovNet.try_gpu()); decoder.eval()
    net = CovNet.Network_Features(6, 10).to(CovNet.try_gpu())
    VAE.load_state_dict(torch.load(data_dir+'Gaussian/network-VAE.params', map_location=CovNet.try_gpu()))
    decoder.load_state_dict(VAE.Decoder.state_dict())
    net.load_state_dict(torch.load(data_dir+"Gaussian/network-latent.params", map_location=CovNet.try_gpu()))

    return decoder, net

def get_covariance(decoder, net, theta, model_vector):
    # first convert theta to the format expected by our emulators
    params = torch.tensor([theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]]).float()
    L = decoder(net(params).view(1,10)).view(100, 100)
    assert not True in torch.isnan(L)
    L = CovNet.symmetric_exp(L).detach().numpy()
    C = np.matmul(L, L.T)
    # C_NG = CovNet.symmetric_exp(L).detach().numpy()
    # C_G = CovaPT.get_gaussian_covariance(params, model_vector)
    # C = C_G + C_NG

    # C_G = CovaPT.get_gaussian_covariance(theta, model_vector)
    # C_SSC = CovaPT.get_SSC_covariance(theta)
    # C = C_G + C_SSC

    try:
        L_2 = np.linalg.cholesky(C)
    except:
        print("ERROR! Matrix is not positive definite, exiting...")
        assert 1 == 0, str(np.amin(C)) + ", " + str(np.amax(C)) + "," + str(params)

    #C = CovaPT.get_gaussian_covariance(params, Pk_galaxy=model_vector)
    return np.linalg.inv(C)

# def get_covariance(Emu, theta):
#     # first convert theta to the format expected by our emulators
#     params = np.array([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]])
#     C = Emu.predict(params)
#     return C

def sample_prior():
    """
    Draws a random starting point based on the prior and returns it
    """
    nParams = cosmo_prior.shape[0] # the number of parameters
    theta0 = np.zeros((nParams))
    for i in range(nParams):
        theta0[i] = (cosmo_prior[i,1]-cosmo_prior[i,0])* np.random.rand(1) + cosmo_prior[i,0] # randomly choose a value in the acceptable range
    
    return theta0 

def model_vector(params, gparams, pgg):
    """
    Calculates the model vector using Yosuke's galaxy power spectrum emulator
    """
    #print(params)
    h = params[0] / 100
    omch2 = params[1]
    ombh2 = params[2]
    #assert omch2 <= 0.131780
    As = params[3]
    #assert As >= 2.47520
    ns = 0.965
    Om0 = (omch2 + ombh2 + 0.00064) / (h**2)
    
    # rebuild parameters into correct format (ombh2, omch2, 1-Om0, ln As, ns, w)
    cparams = np.array([ombh2, omch2, 1-Om0, As, ns, -1])
    redshift = 0.5
    k = np.linspace(0.005, 0.25, 50)
    mu = np.linspace(0.1,0.9,4)
    alpha_perp = 1.1
    alpha_para = 1

    pgg.set_cosmology(cparams, redshift) # <- takes ~0.17s to run
    pgg.set_galaxy(gparams)
    # takes ~0.28 s to run
    P0_emu = pgg.get_pl_gg_ref(0, k, alpha_perp, alpha_para, name='total')
    P2_emu = pgg.get_pl_gg_ref(2, k, alpha_perp, alpha_para, name='total')
    return np.concatenate((P0_emu, P2_emu))

def model_vector_CLASS_PT(params):
    z = 0.58
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

    # This line is necesary to prevent memory leaks
    cosmo.struct_cleanup()
    return [pk_g0, 0, pk_g2, 0, pk_g4]
    #return np.concatenate((pk_g0, pk_g2))

def ln_prior(theta):
    for i in range(len(theta)):
        if (theta[i] < cosmo_prior[i,0]) or (theta[i] > cosmo_prior[i,1]):
            return -np.inf
    return 0.

def ln_lkl(theta, data_vector, decoder, net):
    #with io.StringIO() as buf, redirect_stdout(buf):
    #    x = data_vector - model_vector(theta, gparams, pgg)
    model_vector = model_vector_CLASS_PT(theta)
    x = data_vector - np.concatenate((model_vector[0], model_vector[2]))
    P = get_covariance(decoder, net, theta, model_vector) if vary_covariance == True else P_fid
    lkl = -0.5 * (x.T @ P @ x)
    assert lkl < 0
    return lkl

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data_vector, decoder, net):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data_vector = data_vector
        self.decoder = decoder
        self.net = net

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.data_vector, self.decoder, self.net)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

def main():

    if len(sys.argv) != 3:
        print("USAGE: python Likelihood.py <vary covariance> <resume>")
        return
    vary_covariance = True if sys.argv[1].lower() == "true" else False
    resume = True if sys.argv[2].lower() == "true" else False

    print("Running MCMC with varying covariance: " + str(vary_covariance))
    print("Running with T0 Covariance term: " + str(use_T0))
    if resume == True:
        print("Resuming previous mcmc run...")

    N    = 60000
    NDIM = 6

    # Make sure that the covariance matrix we're using is positive definite
    if vary_covariance == False:
        try:
            L = np.linalg.cholesky(P_fid)
        except np.linalg.LinAlgError as err:
            print("ERROR! Fiducial percision matrix is not positive definite! Exiting...")
            return
        eigen, _ = np.linalg.eig(P_fid)
        if np.all(eigen >= 0.) == False:
            print("ERROR! Fiducial percision matrix is not positive definite! Exiting...")
            return
        if np.all(P_fid != np.nan) == False:
            print("ERROR: NaN values in the percision matrix! Exiting...")
            return

    # Setup data vector as the Pk at the fiducial cosmology
    data_vector = model_vector_CLASS_PT(cosmo_fid)
    data_vector = np.concatenate((data_vector[0], data_vector[2]))
    #data_vector = model_vector(cosmo_fid, gparams, pgg)

    #Cov_emu = PCA_emulator()
    decoder, net = CovNet_emulator()

    # 3ach std is 1/100 times the length of the prior
    #theta_std  = np.array([0.1, 0.00025, 0.000025, 0.0125, 0.005, 0.04])
    #theta_std  = np.array([0.17, 0.00035, 0.00004, 0.02, 0.008, 0.065])
    #C_theta = np.diag(theta_std)**2

    # The parameter covariance matrix is calculated first by assigning it to an arbitrary value 
    # and then running a small chain, after which you calculate it again
    C_theta = [[ 3.962143990721337, -0.004385610849896623, -0.00030540286189768595, -0.10241062772882889, 0.006924269348239488, 0.32942287451610086, ],
              [ -0.004385610849896623, 2.6515841332851822e-05, 4.974629781472943e-07, 0.00018807264719907904, -0.00024509323162060886, -0.0009761931827464128, ],
              [ -0.00030540286189768595, 4.974629781472943e-07, 3.633007555593623e-07, 1.3059893466482729e-05, -1.8930446330439895e-06, -2.247309703905299e-05, ],
              [ -0.10241062772882889, 0.00018807264719907904, 1.3059893466482729e-05, 0.007296053413646783, -0.005173234103352741, -0.015881071905738654, ],
              [ 0.006924269348239488, -0.00024509323162060886, -1.8930446330439895e-06, -0.005173234103352741, 0.006823283642342352, 0.011983121703152919, ],
              [ 0.32942287451610086, -0.0009761931827464128, -2.247309703905299e-05, -0.015881071905738654, 0.011983121703152919, 0.056923623813101225, ]]

    if vary_covariance == True:
        save_str = "pyMC_chains_G_emulate_Varied.npz"
    else: 
        #theta_std  = np.array([0.5, 0.001, 0.0001, 0.1, 0.05, 0.1])
        if use_T0 == True:
            save_str = "pyMC_chains_Fixed.npz"
        else:
            save_str = "pyMC_chains_no_T0.npz"

    print(save_str)

    # Starting position of the emcee chain
    #theta0 = cosmo_fid + (theta_std * np.random.normal(size=(NDIM)))
    theta0 = sample_prior()
    
    logl = LogLike(ln_lkl, data_vector, decoder, net)

    labels = {"H0": ["omch2", "ombh2", "As", "b1", "b2"]}
    MCMC_model = pm.Model()
    with MCMC_model as model:
        H0 = pm.Uniform("H0", cosmo_prior[0,0], cosmo_prior[0,1])
        omch2 = pm.Uniform("omch2", cosmo_prior[1,0], cosmo_prior[1,1])
        ombh2 = pm.Uniform("ombh2", cosmo_prior[2,0], cosmo_prior[2,1])
        As = pm.Uniform("As", cosmo_prior[3,0], cosmo_prior[3,1])
        b1 = pm.Uniform("b1", cosmo_prior[4,0], cosmo_prior[4,1])
        b2 = pm.Uniform("b2", cosmo_prior[5,0], cosmo_prior[5,1])

        theta = tt.as_tensor_variable([H0, omch2, ombh2, As, b1, b2])

        p = pm.DensityDist('p', lambda v: logl(v), observed=(theta))
        step = pm.Metropolis()
        trace = pm.sample(500, step=step, cores=1, chains=1, discard_tuned_samples=False)

    chain = np.vstack((trace["H0"], trace["omch2"], trace["ombh2"], trace["As"], trace["b1"], trace["b2"])).T
    np.savez("Data/"+save_str, chain=chain)

if __name__ == '__main__':
    main()