# This script tests that your enviornment is setup correctly to run CovNet
import unittest
import numpy as np

from CovNet import CovaPT
from CovNet.config import CovaPT_data_dir, CovNet_config_dir
import os

# --------------------------------------------------
# Import tests
# --------------------------------------------------
class TestEnviornment(unittest.TestCase):

    # --------------------------------------------------
    # Compatability tests
    # --------------------------------------------------
    # test that file paths poinitng to external stuff exists
    def test_directories(self):

        # assert directory exits
        print(CovaPT_data_dir)
        self.assertTrue(os.path.exists(CovaPT_data_dir))

        # assert directory is not empty
        #dir = os.listdir(CovaPT_data_dir)
        #self.assertTrue(len(dir) > 0) 

        # make sure the config file directory exists        
        self.assertTrue(os.path.exists(CovNet_config_dir))

        # assert directory is not empty
        # should never be empty because I'm saving a config file to the repo!
        dir = os.listdir(CovNet_config_dir)
        self.assertTrue(len(dir) > 0) 

    # # test wether or not your machine is configured to use pytorch on a gpu
    # if torch.cuda.is_available() == True:
    #     print("Pytorch is configured to run on GPU!")
    # elif torch.backends.mps.is_available() == True:
    #     print("Pytorch is configured to run on M1/2 mac GPU")
    # else:
    #     print("Pytorch is configured to run only on CPU")

    # --------------------------------------------------
    # File-path tests
    # --------------------------------------------------

if __name__ == '__main__':
    unittest.main()