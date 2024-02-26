import unittest
import numpy as np

from CovNet import CovaPT
from CovNet.config import CovaPT_data_dir

class TestCovaPT(unittest.TestCase):

    def test_constructor(self):
        """
        Test the LSS model constructor
        """
        Analytic_Model = CovaPT.LSS_Model(0.61)

        self.assertEqual(Analytic_Model.z, 0.61)

    def test_boltzman_solver(self):
        Analytic_Model = CovaPT.LSS_Model(0.61)
        params = np.array([67.77, 0.1184, 3.0447, 2., 0., 0., 0., 0., 500, 0.])
        output = Analytic_Model.Pk_CLASS_PT(params)

        self.assertNotEqual(output, [])

    def test_covapt(self):
        """
        Test the analytical covariance code with settings I know should work
        """
        Analytic_Model = CovaPT.LSS_Model(0.61)
        params = np.array([67.77, 0.1184, 3.0447, 2., 0., 0., 0., 0., 500, 0.])
        C = Analytic_Model.get_gaussian_covariance(params)

        self.assertFalse(np.any(np.isnan(C)))
        self.assertEqual(len(C), 50)

if __name__ == '__main__':
    unittest.main()