import numpy as np
from numpy import pi, cos
import pymultinest
import threading, subprocess
import torch
import sys, os, io, time, math
from contextlib import redirect_stdout
sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet

training_dir = "/home/joeadamo/Research/CovNet/Data/Training-Set/"
data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
BOSS_dir = "/home/joeadamo/Research/Data/BOSS-DR12/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"

# probability function, taken from the eggbox problem.

C_fixed = np.loadtxt("/home/joeadamo/Research/Data/CovA-survey.txt")

vary_covariance = True

pgg = pkmu_hod()

def show(filepath): 
	""" open the output (pdf) file for the user """
	if os.name == 'mac' or sys.platform == 'darwin': subprocess.call(('open', filepath))
	elif os.name == 'nt' or sys.platform == 'win32': os.startfile(filepath)
	elif sys.platform.startswith('linux') : subprocess.call(('xdg-open', filepath))

cosmo_prior = np.array([[66.5, 75.5],
                        [0.10782, 0.13178],
                        [0.0211375, 0.0233625],
                        [2.4752, 3.7128],#[1.1885e-9, 2.031e-9],
                        [1.806, 2.04],
                        [-2.962, 0.458]])

#                     H0,  omch2, ombh2,  As,  b1,     b2
#cosmo_fid = np.array([69.0,0.1198,0.02225,2e-9,1.9485,-0.5387])
# fiducial values taken from the Patchy paper https://arxiv.org/pdf/1509.06400.pdf
#                     H0,  omch2, ombh2,  As,  b1,     b2
cosmo_fid = np.array([67.8,0.1190,0.02215,2e-9,2.01,-0.47])

gparams = {'logMmin': 13.9383, 'sigma_sq': 0.7918725**2, 'logM1': 14.4857, 'alpha': 1.19196,  'kappa': 0.600692, 
          'poff': 0.0, 'Roff': 2.0, 'alpha_inc': 0., 'logM_inc': 0., 'cM_fac': 1., 'sigv_fac': 1., 'P_shot': 0.}
redshift = 0.58

P0_mean_ref = np.loadtxt(CovaPT_dir+'P0_fit_Patchy.dat')
P2_mean_ref = np.loadtxt(CovaPT_dir+'P2_fit_Patchy.dat')
data_vector = np.concatenate((P0_mean_ref, P2_mean_ref))

def PCA_emulator():
    """
    Sets up the PCA covariance matrix emulator to use in likelihood analysis
    """    
    N_C = 100
    C_PCA = np.zeros((N_C, 100, 100))
    params_PCA = np.zeros((N_C, 6))
    for i in range(N_C):
        temp = np.load(PCA_dir+"CovA-"+f'{i:04d}'+".npz")
        params_PCA[i] = np.delete(temp["params"], 4)
        C_PCA[i] = torch.from_numpy(temp["C"])
    
        # if the matrix doesn't match the transpose close enough, manually flip over the diagonal
        try:
            np.testing.assert_allclose(C_PCA[i], C_PCA[i].T, err_msg="covariance must match transpose")
        except AssertionError:
            L = np.tril(C_PCA[i])
            U = np.tril(C_PCA[i], k=-1).T
            C_PCA[i] = L + U
    Emu = ce.CovEmu(params_PCA, C_PCA, NPC_D=20, NPC_L=20)
    return Emu

def CovNet_emulator():
    """
    Returns a neural net covariance matrix emulator
    """
    VAE = CovNet.Network_VAE().to(CovNet.try_gpu());       VAE.eval()
    decoder = CovNet.Block_Decoder().to(CovNet.try_gpu()); decoder.eval()
    net = CovNet.Network_Features(6, 10).to(CovNet.try_gpu())
    VAE.load_state_dict(torch.load(data_dir+'Correlation-decomp/network-VAE.params', map_location=CovNet.try_gpu()))
    decoder.load_state_dict(VAE.Decoder.state_dict())
    net.load_state_dict(torch.load(data_dir+"Correlation-decomp/network-features.params", map_location=CovNet.try_gpu()))

    return decoder, net

# Define these as global variables so that we don't need to pass them into ln_lkl
decoder, net = CovNet_emulator()
Cov_emu = PCA_emulator()

def model_vector(params, gparams, pgg):
    """
    Calculates the model vector using Yosuke's galaxy power spectrum emulator
    """
    #print(params)
    h = params[0] / 100
    omch2 = params[1]
    ombh2 = params[2]
    #assert omch2 <= 0.131780
    #As = np.log(1e10 * params[3])
    As = params[3]
    #assert As >= 2.47520
    ns = 0.965
    Om0 = (omch2 + ombh2 + 0.00064) / (h**2)
    
    # rebuild parameters into correct format (ombh2, omch2, 1-Om0, ln As, ns, w)
    cparams = np.array([ombh2, omch2, 1-Om0, As, ns, -1])
    redshift = 0.58
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

# def get_covariance(decoder, net, theta):
#     # first convert theta to the format expected by our emulators
#     params = torch.tensor([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]]).float()
#     features = net(params); C = decoder(features.view(1,10)).view(100, 100)
#     C = CovNet.corr_to_cov(C).cpu().detach().numpy()
#     return C

def get_covariance(Emu, theta):
    # first convert theta to the format expected by our emulators
    params = np.array([theta[0], theta[2], theta[1], theta[3], theta[4], theta[5]])
    C = Emu.predict(params)
    return C

def prior(cube, ndim, nparams):
    for i in range(6):
        cube[i] = cube[i] * (cosmo_prior[i, 1] - cosmo_prior[i, 0]) + cosmo_prior[i, 0]

def ln_lkl(cube, ndim, nparams):
    #C = get_covariance(decoder, net, cube) if vary_covariance else C_fixed
    C = get_covariance(Cov_emu, cube) if vary_covariance else C_fixed
    P = np.linalg.inv(C)
    with io.StringIO() as buf, redirect_stdout(buf):
        x = data_vector - model_vector(cube, gparams, pgg)
    lkl = -0.5 * np.matmul(x.T, np.matmul(P, x))
    if lkl > 0:
        print("WARNING: lkl =", lkl, "params:", cube[0], cube[1], cube[2], cube[3], cube[4], cube[5])
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError as err:
        print("Covariance matrix is NOT positive-definite!, lkl =", lkl)
    return lkl

def main():
    # number of dimensions our problem has
    print("Running MCMC with varying covariance: " + str(vary_covariance))

    parameters = ["H0", "omch2","ombh2", "As","b1","b2"]
    n_params = len(parameters)
    # name of the output files
    if vary_covariance == True:
        prefix = "chains/Varied-"
    else: 
        prefix = "chains/Fixed-"

    # https://arxiv.org/pdf/0809.3437.pdf
    t1 = time.time()
    #progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename=prefix); progress.start()
    #threading.Timer(2, show, [prefix+"phys_live.points.pdf"]).start() # delayed opening
    # run MultiNest
    pymultinest.run(ln_lkl, prior, n_params, outputfiles_basename=prefix, 
                    sampling_efficiency = 1, n_live_points=400, resume=False, verbose=True)
    #progress.stop()
    t2 = time.time()
    print("Done!, took {:0.0f} minutes {:0.2f} seconds".format(math.floor((t2 - t1)/60), (t2 - t1)%60))

if __name__ == "__main__":
    main()