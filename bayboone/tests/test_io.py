from unittest import TestCase
from bayboone.data_io import Data
import random
import numpy as np
import os

class TestIo(TestCase):
    
    def test_is_Data_class(self):
        """
        Tests that Data is in fact a Data class
        """
        data = Data(10,5, 1.0)
        assert isinstance(data, Data)
    
    def test_values_stored_correctly(self):
        """
        Tests that values are stored correctly in the class
        """
        data = Data(10, 5,1.0)
        assert data.N_numu == 10
        assert data.N_nue == 5
        assert data.E == 1.0
        
    def test_write_read(self):
        """
        Tests the that write_data and load funcion correctly
        """
        data = Data(10,5,1.0)
        data.write_data('test_data.csv')
        data = Data.load('test_data.csv') 
        assert data.N_numu[0] == 10
        assert data.N_nue[0] == 5
        assert data.E[0] == 1.0
        
    def test_simulate_data(self):
        """
        Tests simulated data with a certain seed. Currently can't get
        the seed to actually be consistent, so for now just testing they are
        integers.
        """
        np.random.seed(28)
        N_nue = Data.simulate_data(Data, N_numu=10, ss2t=1.0, dms=1.0, mu_L=0.5, mu_E=1.0, sigma_L=.025, sigma_E=0.25)
        assert isinstance(N_nue, int)
        
    #def test_zero_oscillation(self):
    #    """
    #    Tests the cases where there should be zero oscillation (N_nue=0).
    #    """
    #    data = Data.simulate_microboone(10, 0.0, 1.0)
    #    assert data.N_nue == 0
    #    
    #    data = Data.simulate_microboone(10, 1.0, 0.0)
    #    assert data.N_nue == 0
    #    
    #    data = Data.simulate_microboone(10, 0.0, 0.0)
    #    assert data.N_nue == 0
        
    def test_all_oscillate(self):
        """
        Test the case where all the neutrinos should oscillate.
        """
        data = Data.simulate_detector(1.0, np.pi/(1.27*2.), 10, 1.0, 1.0, 0.0)
        assert data.N_nue == 10
        
    def test_no_data(self):
        """
        Test input boundaries for zero data points
        """
        data = Data.simulate_detector(1, 1.0, 0, 0.0)
        assert data.N_numu == 0
        assert data.N_nue == 0

        
        
        
      
        
