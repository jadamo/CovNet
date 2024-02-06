import unittest
import os

from CovNet import Emulator
from CovNet.Dataset import load_config_file
from CovNet.config import CovNet_config_dir

class TEstNetwork(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()