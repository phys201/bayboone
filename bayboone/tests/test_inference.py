from bayboone.inference.prior import UniformPrior, JefferysPrior
from bayboone.inference.inference_functions import chisq
from bayboone.data_io import Data

import numpy as np

import unittest
from unittest import TestCase


class TestPriors(TestCase):
    def test_uniform(self):
        assert np.allclose(np.exp(UniformPrior(3, 5).logp(4)), .5)

    def test_jefferys(self):
        assert np.allclose(np.exp(JefferysPrior(10, 1000).logp(100)),
                           0.0021714724095162588)
    
#class TestInference_function(TestCase):
#    def test_chisq(self):
#        data = generative_model(10)
#        assert np.sum(chisq(data, data)) == 0.0
    
if __name__ == '__main__':
    unittest.main()
