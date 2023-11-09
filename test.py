# This script tests that your enviornment is setup correctly to run CovNet

# --------------------------------------------------
# Import tests
# --------------------------------------------------
print("Checking modules can be imported...")
bad_imports = 0
try:
    from easydict import EasyDict
    print("EasyDict: Success")
except:
    print("ERROR! Could not import EasyDict!")
    bad_imports += 1
try:
    import torch
    print("PyTorch: Success")
except:
    print("ERROR! Could not import PyTorch!")
try:
    from classy import Class
    print("CLASS-PT: Success")
except:
    print("ERROR! CLASS-PT was not built correctly! Please follow the instructions at https://github.com/Michalychforever/CLASS-PT/blob/master/instructions.pdf")
    bad_imports += 1

print(bad_imports, "/ 3 modules succesfully imported")

if bad_imports != 0:
    print("To run the rest of these tests, please fix the necesary modules")
else:
    print("yay!")

# --------------------------------------------------
# Compatability tests
# --------------------------------------------------

import src as CovNet