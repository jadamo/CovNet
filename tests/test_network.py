import unittest
import os
import numpy as np

from CovNet import Emulator
from CovNet.Dataset import load_config_file
from CovNet.config import CovNet_config_dir

class TestNetwork(unittest.TestCase):

    def test_emulator_constructor(self):

        example_config = load_config_file(CovNet_config_dir+"covnet_BOSS_hpc.yaml")
        self.assertIsNotNone(example_config)

        test_net = Emulator.Network_Emulator(example_config)
        self.assertIsNotNone(test_net)

    def test_network_settings(self):

        example_config = load_config_file(CovNet_config_dir+"covnet_BOSS_hpc.yaml")
        test_net = Emulator.Network_Emulator(example_config)
        self.assertIsNotNone(test_net)

        example_config.architecture = "MLP"
        test_net_2 = Emulator.Network_Emulator(example_config)
        self.assertIsNotNone(test_net_2)

    def test_emulator_wrapper(self):
        
        test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        test_dir+="/emulators/boss_highz_ngc/MLP-T/"

        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(len(os.listdir(test_dir)) > 0) 

        cov_emulator = Emulator.CovNet(test_dir)
        test_params = np.array([67.77, 0.1184, 3.0447, 2., 0., 0.])

        C_test = cov_emulator.get_covariance_matrix(test_params)

        # test that the resulting matrix behaves well
        self.assertFalse(np.any(np.isnan(C_test)))
        self.assertEqual(len(C_test), 50)
        L = np.linalg.cholesky(C_test)

if __name__ == '__main__':
    unittest.main()