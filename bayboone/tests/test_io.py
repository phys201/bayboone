from unittest import TestCase
from bayboone.data_io import Data
import random
import numpy as np

class TestIo(TestCase):
    
    def test_is_Data_class(self):
        data = Data(10,5)
        assert isinstance(data, Data)
    
    def test_values_stored_correctly(self):
        data = Data(10, 5)
        assert data.N_numu == 10
        assert data.N_nue == 5
        
    def test_write_read(self):
        data = Data(10,5)
        data.write_data('testing.csv')
        data = Data.load('testing.csv')
        assert data.N_numu == 10
        assert data.N_nue == 5
        
    def test_simulate_data(self):
        random.seed(28)
        data = Data.simulate_microboone(10, 1.0, 1.0)
        assert data.N_numu == 10
        assert data.N_nue == 5
        
    def test_zero_oscillation(self):
        data = Data.simulate_microboone(10, 0.0, 1.0)
        assert data.N_nue == 0
        
        data = Data.simulate_microboone(10, 1.0, 0.0)
        assert data.N_nue == 0
        
        data = Data.simulate_microboone(10, 0.0, 0.0)
        assert data.N_nue == 0
        
    def test_all_oscillate(self):
        data = Data.simulate_detector(10, 1.0, np.pi/(1.27*2.), 1.0, 1.0, 0.0, 0.0)
        assert data.N_nue == 10
        
    def test_no_data(self):
        data = Data.simulate_microboone(0, 1.0, 1)
        assert data.N_numu == 0
        assert data.N_nue == 0
        
        
        
      
        
