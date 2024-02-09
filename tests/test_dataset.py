import unittest
import os
import numpy as np

import torch
from CovNet import Dataset
from CovNet.Dataset import load_config_file
from CovNet.config import CovNet_config_dir

class TestDataset(unittest.TestCase):

    def test_normalization(self):
        mat_test = torch.rand(1, 50, 50)

        mat_norm = Dataset.symmetric_log(mat_test, 1., 1.)
        mat_reconstruct = Dataset.symmetric_exp(mat_norm, 1., 1.)

        diff = mat_reconstruct - mat_test

        self.assertTrue(torch.max(diff).item() < 2e-7)

    def test_rearangement(self):
        # test with a random lower-triangular matrix
        mat_test = torch.rand(1, 50, 50)
        mat_test = torch.tril(mat_test)

        mat_half = Dataset.rearange_to_half(mat_test, 50)
        mat_full = Dataset.rearange_to_full(mat_half, 50, False)

        self.assertEqual(mat_full.shape, mat_test.shape)
        
        diff = mat_test - mat_full
        self.assertTrue(torch.max(diff).item() == 0)


if __name__ == '__main__':
    unittest.main()