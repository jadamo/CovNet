# simple script to combine individual matrix files into training / validation / test sets
# Due to how I generate my training set, an incomplete set might have missing indices
# (ex: [cov-1111, cov-1113], so 1112 is missing). In order to deal with this, run this script
# to remove index gaps
import os
import numpy as np
from tqdm import tqdm

def main():

    in_path = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC/"
    out_path = "/xdisk/timeifler/jadamo/Training-Set-HighZ-NGC/"
    #path = "/home/joeadamo/Research/CovNet/Data/Inportance-Set/"
    #N = 150400
    N = 1052800
    idx = 0

    train_frac = 0.8
    valid_frac = 0.1
    test_frac = 0.1

    params = np.zeros((N, 6))
    C_G = np.zeros((N, 50, 50))
    C_NG = np.zeros((N, 50, 50))
    remove = 0
    total = 0

    # load in all the matrices internally (NOTE: memory intensive!)
    for i in tqdm(range(N)):
        filename = in_path + "CovA-"+f'{i:06d}'+".npz"
        if os.path.exists(filename):
            F_1 = np.load(filename)
            try:
                L = np.linalg.cholesky(F_1["C_G"] + F_1["C_NG"])
                #L2 = np.linalg.cholesky(F_1["C_G"])

                params[i] = F_1["params"]
                C_G[i] = F_1["C_G"]
                C_NG[i] = F_1["C_NG"]
                total+=1

            except:
                os.remove(filename)
                remove += 1
    
    print("Removed", remove, "files for being non positive definite")
    print("Found", total, "files")

    idx = np.where(params[:,0] != 0)
    params = params[idx]
    C_G = C_G[idx]
    C_NG = C_NG[idx]

    N = params.shape[0]

    N_train = int(N * train_frac)
    N_valid = int(N * valid_frac)
    N_test = int(N * test_frac)
    assert N_train + N_valid + N_test <= N

    valid_start = N_train
    valid_end = N_train + N_valid
    test_end = N_train + N_valid + N_test
    assert test_end - valid_end == N_test
    assert valid_end - valid_start == N_valid

    print("splitting dataset into chunks of size [{:0.0f}, {:0.0f}, {:0.0f}]...".format(N_train, N_valid, N_test))

    np.savez(out_path+"CovA-training.npz", 
             params=params[0:N_train], C_G=C_G[0:N_train], C_NG=C_NG[0:N_train])
    np.savez(out_path+"CovA-validation.npz", 
             params=params[valid_start:valid_end], C_G=C_G[valid_start:valid_end], C_NG=C_NG[valid_start:valid_end])
    np.savez(out_path+"CovA-testing.npz", 
             params=params[valid_end:test_end], C_G=C_G[valid_end:test_end], C_NG=C_NG[valid_end:test_end])    

    print("Done!")
    # for i in range(N):
    #     filename = path + "CovA-"+f'{i:05d}'+".npz"
    #     # use a different prefix to prevent overwriting files
    #     new_filename = path + "Cov-"+f'{idx:05d}'+".npz"
    #     if os.path.exists(filename):
    #         os.rename(filename, new_filename)
    #         idx+= 1

    # # rewrite everything to the original prefix
    # for j in range(N):
    #     filename = path + "Cov-"+f'{j:05d}'+".npz"
    #     new_filename = path + "CovA-"+f'{j:05d}'+".npz"
    #     if os.path.exists(filename):
    #         os.rename(filename, new_filename)

    # print("Done! renamed " + str(idx) + " files")

if __name__ == "__main__":
    main()
