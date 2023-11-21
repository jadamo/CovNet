# This script tests that your enviornment is setup correctly to run CovNet

# --------------------------------------------------
# Import tests
# --------------------------------------------------
print("Checking modules can be imported...")
success_imports = 0
num_tests = 4

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
    from classy import Class
    print("CLASS-PT: Success")
    success_imports+= 1
except:
    print("ERROR! CLASS-PT was not built correctly! Please follow the instructions at https://github.com/Michalychforever/CLASS-PT/blob/master/instructions.pdf")


print(success_imports, "/", num_tests, "modules succesfully imported")

if success_imports != num_tests:
    print("To run the rest of these tests, please fix the necesary modules")
else:
    print("yay!")

# --------------------------------------------------
# Compatability tests
# --------------------------------------------------

import src as CovNet