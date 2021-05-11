from unittest import TestCase
from bayboone.data_io import Data
import random
import numpy as np

class TestIo(TestCase):
    
    def test_is_Data_class(self):
        """
        Tests that Data is in fact a Data class
        """
        data = Data(10, 5, 1.0)
        assert isinstance(data, Data)
    
    def test_values_stored_correctly(self):
        """
        Tests that values are stored correctly in the class
        """
        data = Data(10, 5, 1.0)
        assert data.N_numu == 10
        assert data.N_nue == 5
        assert data.E == 1.0
        
    def test_write_read(self):
        """
        Tests the that write_data and load funcion correctly
        """
        data = Data(10, 5, 1.0)
        data.write_data('test_data.csv')
        data = Data.load('test_data.csv') 
        assert data.N_numu == 10
        assert data.N_nue == 5
        assert data.E == 1.0
        
class TestSimulateData(TestCase):
        
    def test_simulate_data(self):
        """
        Tests simulated data with a certain seed. Currently can't get
        the seed to actually be consistent, so for now just testing they are
        integers.
        """
        N_nue = Data.simulate_data(Data, N_numu=10, ss2t=1.0, dms=1.0, mu_L=0.5, mu_E=1.0, sigma_L=.025, sigma_E=0.25)
        assert isinstance(N_nue, int)
        
    def test_bad_ss2t(self):
        bad_ss2t_list = [1.1, -0.1]
        dms = 1.0
        N_numu = 10
        E = 1.0
        L = .5
        sigma_L = .1
        sigma_E = .1
        for bad_ss2t in bad_ss2t_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, bad_ss2t, dms, L, E, sigma_L, sigma_E)
    
    def test_bad_dms(self):
        ss2t = 0.5
        bad_dms_list = [-.1]
        N_numu = 10
        L = .5
        E = 1.0
        sigma_L = .1
        sigma_E = .1
        for bad_dms in bad_dms_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, ss2t, bad_dms, L, E, sigma_L, sigma_E)
    
    def test_bad_N_numu(self):
        ss2t = 0.5
        dms = 1.0
        bad_N_numu_list = [-1, 0, 1.5]
        L = .5
        E = 1.0
        sigma_L = .1
        sigma_E = .1
        for bad_N_numu in bad_N_numu_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, bad_N_numu, ss2t, dms, L, E, sigma_L, sigma_E)
            
    def test_bad_L(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        bad_L_list = [-1]
        E = 1.0
        sigma_L = .1
        sigma_E = .1
        for bad_L in bad_L_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, ss2t, dms, bad_L, E, sigma_L, sigma_E)
    
    def test_bad_E(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        L = .5
        bad_E_list = [0.0, -1.0]
        sigma_L = .1
        sigma_E = .1
        for bad_E in bad_E_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, ss2t, dms, L, bad_E, sigma_L, sigma_E)
            
    def test_bad_sigma_L(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        L = .5
        E = 1.0
        bad_sigma_L_list = [-1., 0.]
        sigma_E = .1
        for bad_sigma_L in bad_sigma_L_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, ss2t, dms, L, E, bad_sigma_L, sigma_E)
            
    def test_bad_sigma_E(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        L = .5
        E = 1.0
        sigma_L = .1
        bad_sigma_E_list = [-1., 0.]
        for bad_sigma_E in bad_sigma_E_list:
            self.assertRaises(ValueError, Data.simulate_data, Data, N_numu, ss2t, dms, L, E, sigma_L, bad_sigma_E)

        
class TestSimulateDetector(TestCase):
    
    def test_zero_oscillation(self):
        """
        Tests the cases where there should be zero oscillation (N_nue=0).
        """
        data = Data.simulate_detector(0.0, 1.0, 10, 1.0)
        assert data.N_nue == 0
    
        data = Data.simulate_detector(1.0, 0.0, 10, 1.0)
        assert data.N_nue == 0
        
        data = Data.simulate_detector(0.0, 0.0, 10, 1.0)
        assert data.N_nue == 0

    
    def test_bad_single_input_ss2t(self):
        bad_ss2t_list = [1.1, -0.1]
        dms = 1.0
        N_numu = 10
        E = 1.0
        for bad_ss2t in bad_ss2t_list:
            self.assertRaises(ValueError, Data.simulate_detector, bad_ss2t, dms, N_numu, E)
            
    def test_bad_single_input_dms(self):
        ss2t = 0.5
        bad_dms_list = [-.1]
        N_numu = 10
        E = 1.0 
        for bad_dms in bad_dms_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, bad_dms, N_numu, E)
            
    def test_bad_single_input_N_numu(self):
        ss2t = 0.5
        dms = 1.0
        bad_N_numu_list = [-1, 0, 1.5]
        E = 1.0
        for bad_N_numu in bad_N_numu_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, bad_N_numu, E)
            
    def test_bad_single_input_E(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        bad_E_list = [0.0, -1.0]
        for bad_E in bad_E_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, N_numu, bad_E)
            
    def test_bad_single_input_L(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        E = 1.
        bad_L_list = [-1.0]
        for bad_L in bad_L_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, N_numu, E, bad_L)  
            
    def test_bad_single_input_sigma_L(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = 10
        E = 1.
        L = .5
        bad_sigma_L_list = [-1.0, 0.0]
        for bad_sigma_L in bad_sigma_L_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, N_numu, E, L, bad_sigma_L)
        
    def test_bad_array_input_N_numu(self):
        ss2t = 0.5
        dms = 1.0
        E_bin_edges = [1.0, 2.0, 3.0]
        bad_N_numu_list = [[-1,1], [0,1], [1.5,1], [1]]
        for bad_N_numu in bad_N_numu_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, bad_N_numu, E_bin_edges)
        
    def test_bad_array_input_E(self):
        ss2t = 0.5
        dms = 1.0
        N_numu = [10,10]
        E_bin_edges = [1.0, 2.0, 3.0]
        bad_E_bin_edges_list = [[0.0, 0.0, 2.0], [-1.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 2.0]]
        for bad_E_bin_edges in bad_E_bin_edges_list:
            self.assertRaises(ValueError, Data.simulate_detector, ss2t, dms, N_numu, bad_E_bin_edges)


