import unittest
import numpy as np

from CovNet import window
from CovNet.config import CovaPT_data_dir

class TestWindowKernels(unittest.TestCase):

    def test_gaussian_window_kernel_constructor(self):
        """
        Test the window kernel class constructor, as well as some
        of its helper functions
        """
        k_centers_test = np.linspace(0.005, 0.245, 25)
        k_width_test = 0.01
        k_edges_test = np.linspace(0, 0.25, 26)
        
        Window_Kernels = window.Gaussian_Window_Kernels(k_centers_test)
        self.assertAlmostEqual(k_width_test, Window_Kernels.kbin_width)
        
        self.assertEqual(len(k_edges_test), len(Window_Kernels.kbin_edges))
        diff = k_edges_test - Window_Kernels.kbin_edges
        max_diff = np.amax(diff)
        self.assertAlmostEqual(max_diff, 0)


if __name__ == '__main__':
    unittest.main()