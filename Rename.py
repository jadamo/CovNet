# simple script to rename files in an incomplete training set
# Due to how I generate my training set, an incomplete set might have missing indices
# (ex: [cov-1111, cov-1113], so 1112 is missing). In order to deal with this, run this script
# to remove index gaps
import os
import numpy as np

def main():

    path = "/home/u12/jadamo/CovNet/Training-Set-HighZ-NGC-old/"
    N = 150400
    idx = 0

    remove = 0
    for i in range(N):
        filename = path + "CovA-"+f'{i:05d}'+".npz"
        if os.path.exists(filename):
            F_1 = np.load(filename)
            C = F_1["C_G"] + F_1["C_SSC"] + F_1["C_T0"]
            try:
                L = np.linalg.cholesky(C)
                L2 = np.linalg.cholesky(F_1["C_G"])
            except np.linalg.LinAlgError as err:
                os.remove(filename)
                remove += 1
    
    print("Removed", remove, "files for being non positive definite")

    for i in range(N):
        filename = path + "CovA-"+f'{i:05d}'+".npz"
        # use a different prefix to prevent overwriting files
        new_filename = path + "Cov-"+f'{idx:05d}'+".npz"
        if os.path.exists(filename):
            os.rename(filename, new_filename)
            idx+= 1

    # rewrite everything to the original prefix
    for j in range(N):
        filename = path + "Cov-"+f'{j:05d}'+".npz"
        new_filename = path + "CovA-"+f'{j:05d}'+".npz"
        if os.path.exists(filename):
            os.rename(filename, new_filename)

    print("Done! renamed " + str(idx) + " files")

if __name__ == "__main__":
    main()
