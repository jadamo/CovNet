import unittest
import numpy as np

from CovNet import CovaPT
from CovNet.config import CovaPT_data_dir

class TestCovaPT(unittest.TestCase):

    def test_constructor(self):
        Analytic_Model = CovaPT.Analytic_Covmat(0.61)

        self.assertEqual(Analytic_Model.z, 0.61)

        self.assertEqual(Analytic_Model.dire, CovaPT_data_dir)

    def test_boltzman_solver(self):
        Analytic_Model = CovaPT.Analytic_Covmat(0.61)
        params = np.array([67.77, 0.1184, 3.0447, 2., 0., 0., 0., 0., 500, 0.])
        output = Analytic_Model.Pk_CLASS_PT(params)

        self.assertNotEqual(output, [])

if __name__ == '__main__':
    unittest.main()