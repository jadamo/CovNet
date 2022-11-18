import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
from tqdm import tqdm
from scipy.stats import norm
from classy import Class

sys.path.insert(1, '/home/joeadamo/Research/covariance_emulator')
import covariance_emulator as ce
sys.path.append('/home/joeadamo/Research') #<- parent directory of dark emulator code
from DarkEmuPowerRSD import pkmu_nn, pkmu_hod
import CovNet, CovaPT
sys.path.append('/home/joeadamo/Research/Software')
from pk_tools import pk_tools

data_dir =  "/home/joeadamo/Research/CovNet/Data/"
PCA_dir = "/home/joeadamo/Research/CovNet/Data/PCA-Set/"
CovaPT_dir = "/home/joeadamo/Research/CovaPT/Example-Data/"
BOSS_dir = "/home/joeadamo/Research/Data/BOSS-DR12/Updated/"

use_T0 = True

C_fid_file = np.load(data_dir+"Cov_Fid.npz")
if use_T0 == True:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"] + C_fid_file["C_T0"]
else:
    C_fid = C_fid_file["C_G"] + C_fid_file["C_SSC"]

P_fid = np.linalg.inv(C_fid)

cosmo_prior = np.array([[52, 100],     #H0
                        [0.002, 0.3],  #omch2
                        [0.005, 0.08], #ombh2
                        [0.3, 1.6],    #A / A_planck
                        [0.9, 1.1],    #ns
                        [1, 4],        #b1
                        [0, 1],        #b2 (gaussian)
                        [0, 1],        #bGamma2 (gaussian)
                        [0, 30],       #c0 (gaussian)
                        [0, 30],       #c2 (gaussian)
                        [0, 5000]      #Pshot (gaussian)
                        ])

A_planck = 3.0448

# fiducial taken to be the cosmology used to generate Patchy mocks
#                     H0,   omch2,  ombh2,  A,     ns,     b1,     b2      bG2, c0, c2,  Pshot
#cosmo_fid = np.array([67.77,0.11827,0.02214,1.016, 0.9611, 1.9640,-0.5430, 0.1, 5., 15., 5e3])
cosmo_fid = np.array([67.77,0.11827,0.02214,1.016, 0.9611, 1.9640, 0., 0., 0., 0., 0])

redshift = 0.61

W = pk_tools.read_matrix(BOSS_dir+"W_CMASS_North.matrix")
M = pk_tools.read_matrix(BOSS_dir+"M_CMASS_North.matrix")

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
        if i < 6:
            theta0[i] = (cosmo_prior[i,1]-cosmo_prior[i,0])* np.random.rand(1) + cosmo_prior[i,0] # randomly choose a value in the acceptable range
        else:
            theta0[i] = np.random.normal(cosmo_prior[i,0], cosmo_prior[i,1])

    return theta0 

def model_vector_CLASS_PT(params):
    z = 0.61

    As = params[3] * A_planck

    cosmo = Class()
    cosmo.set({'output':'mPk',
            'non linear':'PT',
            'IR resummation':'Yes',
            'Bias tracers':'Yes',
            'cb':'Yes', # use CDM+baryon spectra
            'RSD':'Yes',
            'AP':'Yes', # Alcock-Paczynski effect
            'Omfid':'0.307115', # fiducial Omega_m
            'PNG':'No', # single-field inflation PNG
            'FFTLog mode':'FAST',
            'A_s':np.exp(As)/1e10,
            'n_s':params[4],
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
    k = np.linspace(0.005, 0.395, 400)
    cosmo.compute()
    cosmo.initialize_output(k, z, len(k))

    b1 = params[5] / np.sqrt(params[3])
    b2 = params[6] / np.sqrt(params[3])
    bG2 = params[7] / np.sqrt(params[3])
    bGamma3 = 0
    cs4 = -5
    b4 = 100. # from CLASS-PT Notebook (I don't think Wadekar varies this)
    cs0, cs2, Pshot = params[8], params[9], params[10]
    pk_g0 = cosmo.pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot, b4)
    pk_g2 = cosmo.pk_gg_l2(b1, b2, bG2, bGamma3, cs2, b4)
    pk_g4 = cosmo.pk_gg_l4(b1, b2, bG2, bGamma3, cs4, b4)

    # Convolve with the window function
    model_vector = np.concatenate((pk_g0, pk_g2, pk_g4))
    model_vector = np.matmul(M, model_vector)
    model_vector = np.matmul(W, model_vector)

    # This line is necesary to prevent memory leaks
    cosmo.struct_cleanup()

    return np.concatenate([model_vector[0:25], model_vector[80:105]])

def ln_prior(theta):
    prior = 0.
    for i in range(len(theta)):
        if i < 6:
            if (theta[i] < cosmo_prior[i,0]) or (theta[i] > cosmo_prior[i,1]):
                return -np.inf
        else:
            dist = norm(cosmo_prior[i,0], cosmo_prior[i,1])
            prior += np.log(dist.pdf(theta[i]))
    return prior

def ln_lkl(theta, data_vector, decoder, net, vary_covariance):
    #with io.StringIO() as buf, redirect_stdout(buf):
    #    x = data_vector - model_vector(theta, gparams, pgg)
    model_vector = model_vector_CLASS_PT(theta)
    x = data_vector - model_vector
    P = get_covariance(decoder, net, theta, model_vector) if vary_covariance == True else P_fid
    lkl = -0.5 * (x.T @ P @ x)
    assert lkl <= 0
    return lkl

def ln_prob(theta, data_vector, decoder, net, vary_covariance):
    p = ln_prior(theta)
    if p != -np.inf:
        return p + ln_lkl(theta, data_vector, decoder, net, vary_covariance)
    else: return p

def Metropolis_Hastings(theta, C_theta, N, NDIM, data_vector, decoder, net, vary_covariance, resume, save_str):
    """
    runs an mcmc based on metropolis hastings
    """
    if resume == False:
        acceptance_rate = np.zeros(N)
        num_accept = 0
        chain = np.zeros((N, NDIM))
        log_lkl = np.zeros(N)
        start = 0
    else:
        file = np.load("Data/"+save_str)
        chain = file["chain"]
        acceptance_rate = file["rate"]
        log_lkl = file["lkl"]
        start = np.where(acceptance_rate[(acceptance_rate != 0)])[0][-1] + 1
        theta = chain[start]
        num_accept = acceptance_rate[start] * (start+1)
        del file

    prob_old = ln_prob(theta, data_vector, decoder, net, vary_covariance)
    for i in tqdm(range(start, N)):
        # STEP 1: save current state to the chain
        chain[i] = theta
        log_lkl[i] = prob_old

        # STEP 2: generate a new potential move
        #theta_new = (np.random.normal(loc=theta, scale=theta_std, size=(NDIM)))
        theta_new = np.random.multivariate_normal(mean=theta, cov=C_theta)

        # STEP 3: determine if we move to the new position based on ln_prob
        prob_new = ln_prob(theta_new, data_vector, decoder, net, vary_covariance)
        p = np.random.uniform()

        #print(prob_new, np.exp(prob_new))
        if p < min(np.exp(prob_new - prob_old), 1):
            theta = theta_new
            prob_old = prob_new
            num_accept += 1

        acceptance_rate[i] = num_accept / (i+1.)

        if i % 500 == 0:
            np.savez("Data/"+save_str, chain=chain, lkl=log_lkl, rate=acceptance_rate)

    return chain, log_lkl, acceptance_rate

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

    N    = 100000
    NDIM = 11

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

    # Data vector taken from https://fbeutler.github.io/hub/deconv_paper.html
    #pk_dict = pk_tools.read_power(BOSS_dir+"P_CMASS_North.dat" , combine_bins =10)
    #data_vector = np.concatenate([pk_dict["pk0"][:25], pk_dict["pk2"][:25]])
    data_vector = model_vector_CLASS_PT(cosmo_fid)

    # Setup data vector as the Pk at the fiducial cosmology
    #data_vector = model_vector_CLASS_PT(cosmo_fid)
    #data_vector = np.concatenate((data_vector[0], data_vector[2]))

    #Cov_emu = PCA_emulator()
    #decoder, net = CovNet_emulator()
    decoder, net = None, None

    print("min likelihood = ", -2*ln_prob(cosmo_fid, data_vector, decoder, net, vary_covariance))
    last_theta = np.array([9.70979124e+01,  9.25151521e-02,  1.13833147e-02,  8.75263267e-01, 1.09011685e+00,  1.54921836e+00, -2.89903007e-01, -2.07437562e-01, 2.16834442e+01, -1.54617096e+00 , 5.27260532e+03])
    print("last likelihood = ", -2*ln_prob(last_theta, data_vector, decoder, net, vary_covariance))


    # Each naive std is 1/100 times the length of the prior
    # theta_std  = np.array([0.15, 0.0006, 0.00005, 0.015, 0.03, 0.1, 0.2, 
    #                       0.25, 1.5, 1.5, 100.])
    # C_theta = np.diag(theta_std**2)

    # The parameter covariance matrix is calculated first by assigning it to an arbitrary value 
    # and then running a small chain, after which you calculate it again
    C_theta = [[ 0.1697034955221653, 0.0018756485455925004, -0.00012755042936932873, -0.010332333097359852, -0.0033011799151297146, 0.006200354536014686, -0.6436928031008571, -0.08722384596619201, -0.4350313866824416, -5.051284489525018, 256.2780292321567],
            [ 0.0018756485455925004, 2.799370316852879e-05, -1.8453784982465928e-06, -0.00012758186346366332, -1.8352429736795777e-05, 0.0006312252042537553, -0.00801201765737572, -0.0004823754046332929, 0.0020136988499557746, -0.07129976129068936, 2.9971432614722766],
            [ -0.00012755042936932873, -1.8453784982465928e-06, 1.280591606786456e-07, 1.3229372538141804e-05, 1.5038287395589148e-06, -3.961782848568164e-05, 0.0005532574588257828, 6.938278190532095e-05, 3.396951038779364e-05, 0.0041763784899615745, -0.17264736168221614],
            [ -0.010332333097359852, -0.00012758186346366332, 1.3229372538141804e-05, 0.005992943270853318, 0.0002207541695949502, -0.00964197579582706, 0.03595480215171107, 0.036787840412673654, 0.052665129782016276, -0.4277753279325201, 30.26720873385464],
            [ -0.0033011799151297146, -1.8352429736795777e-05, 1.5038287395589148e-06, 0.0002207541695949502, 0.00016282373245790955, 0.0018569112010070313, 0.012363940957063528, 0.0044137118080330015, 0.02034114827824286, 0.05104464045976817, -4.586211763094009],
            [ 0.006200354536014686, 0.0006312252042537553, -3.961782848568164e-05, -0.00964197579582706, 0.0018569112010070313, 0.09335408994943094, 0.027777565011845248, 0.02399491417682217, 0.6542973382321724, 0.23534119915331547, -97.03749370508764],
            [ -0.6436928031008571, -0.00801201765737572, 0.0005532574588257828, 0.03595480215171107, 0.012363940957063528, 0.027777565011845248, 2.949590154943253, 0.35384595482484155, 1.899159794079759, 23.621173375470676, -1222.5351008932134],
            [ -0.08722384596619201, -0.0004823754046332929, 6.938278190532095e-05, 0.036787840412673654, 0.0044137118080330015, 0.02399491417682217, 0.35384595482484155, 0.3294484978231259, 0.75025413955197, -2.9550103209501883, 129.29793022615905],
            [ -0.4350313866824416, 0.0020136988499557746, 3.396951038779364e-05, 0.052665129782016276, 0.02034114827824286, 0.6542973382321724, 1.899159794079759, 0.75025413955197, 21.24247266774503, 7.294219271434999, -1209.5534159524352],
            [ -5.051284489525018, -0.07129976129068936, 0.0041763784899615745, -0.4277753279325201, 0.05104464045976817, 0.23534119915331547, 23.621173375470676, -2.9550103209501883, 7.294219271434999, 310.2678038182029, -15933.406639067536],
            [ 256.2780292321567, 2.9971432614722766, -0.17264736168221614, 30.26720873385464, -4.586211763094009, -97.03749370508764, -1222.5351008932134, 129.29793022615905, -1209.5534159524352, -15933.406639067536, 906646.5149391479]]
    # C_theta = [[ 59.701635739399165, 0.11718366222798778, -0.001575948849060138, 0.3041304679297917, -0.33031454009936423, -4.305132113831403, 2.0446929058559014, -6.673789819878645, -434.0780271117823, 303.9647122578785, -9857.735686069585],
    #         [ 0.11718366222798778, 0.0003568176391651403, -8.445368959949013e-06, -4.217127384876915e-05, -0.0006942631181543851, -0.007631397175413029, 0.004326377651603874, -0.011428819343912149, -0.7870604261643722, 0.5925786732235578, -18.85495913572744],
    #         [ -0.001575948849060138, -8.445368959949013e-06, 6.538688752325068e-07, 3.246833448220497e-05, 1.3541510378974185e-05, 6.939806404177026e-05, 1.479574625616169e-05, -2.3177520903475487e-05, 0.0093011166261074, -0.007844316365046868, 0.19148784119760376],
    #         [ 0.3041304679297917, -4.217127384876915e-05, 3.246833448220497e-05, 0.013745549722847774, -0.001205514745582804, -0.03033738295806196, -0.019629812217130414, -0.01070222528760603, -2.054461647530049, 2.5950001799684848, -18.9273109425686],
    #         [ -0.33031454009936423, -0.0006942631181543851, 1.3541510378974185e-05, -0.001205514745582804, 0.003910562686442331, 0.024723888334977705, -0.01047051646122979, 0.03674771345587102, 2.54550739051394, -1.6779713038358772, 58.81509895838024],
    #         [ -4.305132113831403, -0.007631397175413029, 6.939806404177026e-05, -0.03033738295806196, 0.024723888334977705, 0.38109116756819583, -0.12012856409420239, 0.5111103503101484, 33.337442783185104, -24.58169393077532, 718.6408639323533],
    #         [ 2.0446929058559014, 0.004326377651603874, 1.479574625616169e-05, -0.019629812217130414, -0.01047051646122979, -0.12012856409420239, 1.0775472776950819, -0.4172886162871941, -11.452368740691897, 7.835491256429384, -503.5217428498312],
    #         [ -6.673789819878645, -0.011428819343912149, -2.3177520903475487e-05, -0.01070222528760603, 0.03674771345587102, 0.5111103503101484, -0.4172886162871941, 1.5784106559336368, 42.042921490817974, -25.178371583494325, 1136.7555363063125],
    #         [ -434.0780271117823, -0.7870604261643722, 0.0093011166261074, -2.054461647530049, 2.54550739051394, 33.337442783185104, -11.452368740691897, 42.042921490817974, 3978.8186270733495, -2296.5623218296696, 91152.56349135115],
    #         [ 303.9647122578785, 0.5925786732235578, -0.007844316365046868, 2.5950001799684848, -1.6779713038358772, -24.58169393077532, 7.835491256429384, -25.178371583494325, -2296.5623218296696, 2254.2254386584304, -54712.80399024705],
    #         [ -9857.735686069585, -18.85495913572744, 0.19148784119760376, -18.9273109425686, 58.81509895838024, 718.6408639323533, -503.5217428498312, 1136.7555363063125, 91152.56349135115, -54712.80399024705, 2556836.914669397]]
     

    if vary_covariance == True:
        save_str = "mcmc_chains_G_emulate_Varied.npz"
    else: 
        #theta_std  = np.array([0.5, 0.001, 0.0001, 0.1, 0.05, 0.1])
        if use_T0 == True:
            save_str = "mcmc_chains_Fixed.npz"
        else:
            save_str = "mcmc_chains_no_T0.npz"

    print(save_str)

    # Starting position of the emcee chain
    #theta0 = cosmo_fid + (theta_std * np.random.normal(size=(NDIM)))
    theta0 = sample_prior()

    chain, log_lkl, acceptance_rate = Metropolis_Hastings(theta0, C_theta, N, NDIM, data_vector, decoder, net, vary_covariance, resume, save_str)
    np.savez("Data/"+save_str, chain=chain, lkl=log_lkl, rate=acceptance_rate)

if __name__ == '__main__':
    main()