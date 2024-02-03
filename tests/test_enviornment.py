# This script tests that your enviornment is setup correctly to run CovNet
# TODO: replace this with something more formal like unittest
#import unittest

# --------------------------------------------------
# Import tests
# --------------------------------------------------
def main():

    print("Checking all required packages can be imported...")
    success_imports = 0
    num_tests = 6

    try:
        import numpy as np
        print("NumPy: Success")
        success_imports+= 1
    except:
        print("ERROR! Could not import Numpy!")
    try:
        from easydict import EasyDict
        print("EasyDict: Success")
        success_imports+= 1
    except:
        print("ERROR! Could not import EasyDict!")
    try:
        import torch
        print("PyTorch: Success")
        success_imports+= 1
    except:
        print("ERROR! Could not import PyTorch!")
    try:
        import yaml
        print("yaml: Success")
        success_imports+= 1
    except:
        print("ERROR! Could not import yaml!")

    try:
        import camb
        print("camb: Success")
        success_imports+= 1
    except:
        print("ERROR: Could not import camb!")
    try:
        from classy import Class
        print("CLASS-PT: Success")
        success_imports+= 1
    except:
        print("ERROR! CLASS-PT was not built correctly! Please follow the instructions at https://github.com/Michalychforever/CLASS-PT/blob/master/instructions.pdf")


    print(success_imports, "/", num_tests, "modules succesfully imported")

    if success_imports != num_tests:
        print("To run the rest of these tests, please fix the necesary modules")
        return 0

    # --------------------------------------------------
    # Compatability tests
    # --------------------------------------------------
    from CovNet import CovaPT

    # test wether or not your machine is configured to use pytorch on a gpu
    if torch.cuda.is_available() == True:
        print("Pytorch is configured to run on GPU!")
    elif torch.backends.mps.is_available() == True:
        print("Pytorch is configured to run on M1/2 mac GPU")
    else:
        print("Pytorch is configured to run only on CPU")

    # test that you can use CLASS-PT without triggering a segmentation fault
    # if not, then it's not configured correctly
    Analytic_Model = CovaPT.Analytic_Covmat(0.61)
    params = np.array([67.77, 0.1184, 3.0447, 2., 0., 0., 0., 0., 500, 0.])
    output = Analytic_Model.Pk_CLASS_PT(params)
    if len(output) == 0:
        print("ERROR: Bolztman Solver failed! This is probably due to CLASS-PT being configured incorrectly")
    else:
        print("CLASS-PT configuration test: success!")

    # --------------------------------------------------
    # File-path tests
    # --------------------------------------------------

if __name__ == "__main__":
    main()