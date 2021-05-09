from unittest import TestCase

#Dependencies of our module
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az

from bayboone.inference import module

class TestOscillation_model(TestCase):    
    
    def test_produces_a_model(self):        
        test_model = module.oscillation_model(10, 1)      
        self.assertTrue(isinstance(test_model, pm.model.Model))    
        
    def test_rejects_unresonable_inputs(self):        
        num_numu, num_nue = (10, 22)          
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue)
        
    def test_rejects_unphysical_mixing_params(self):     
        num_numu, num_nue = 100, 1
        ss2t_bad, ss2t_neg = 22, -1         
        dms_bad = -0.01
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, est_ss2t = ss2t_bad)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, est_ss2t = ss2t_neg)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, est_dms = dms_bad)
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, est_ss2t = ss2t_bad, est_dms = dms_bad)    
        
    def test_rejects_unphysical_LE(self):     
        num_numu, num_nue = 100, 1
        L_test, E_test = (-1, -1)
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, E = E_test)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, L = L_test)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, E = E_test, L = L_test)
        
    def test_rejects_unphysical_stdLE(self):     
        num_numu, num_nue = 100, 1
        stdL_test, stdE_test = (-1, -1)
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, std_E = stdE_test)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, std_L = stdL_test)    
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue, std_E = stdE_test, std_L = stdL_test)  
        
        

        
class TestInference(TestCase):    
    #Apparently we might not need these - not deleting for now just in case, but probably will in a later update
    ss2t = 0.1 #unit-less
    dms = 1 #eV^2
    E = 1 #GeV
    L = 0.5 #km
    num_numu = 600000
    num_nue = num_numu*ss2t*np.sin(1.27*dms*L/E)**2
    #trace = module.fit_model(num_numu, num_nue, 1000)
    
    def test_fit_model_guess_dict(self):
        bad_guess = {'el': 0}
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess = bad_guess)
        
    def test_fit_model_guess_values(self):
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'E': -1.0})
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'L': -1.0})
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'sin^2_2theta': -1.0})
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'sin^2_2theta': 1.1})
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'delta_m^2': -1.0})
        self.assertRaises(ValueError, module.fit_model, TestInference.num_numu, TestInference.num_nue, initial_guess ={'rate': -1.0})
                    
            
    def test_rejects_unresonable_inputs(self):        
        num_numu, num_nue = (10, 22)          
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue)
        
    def test_rejects_unphysical_mixing_params(self):     
        num_numu, num_nue = 100, 1
        ss2t_bad, ss2t_neg = 22, -1         
        dms_bad = -0.01
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, est_ss2t = ss2t_bad)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, est_ss2t = ss2t_neg)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, est_dms = dms_bad)
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, est_ss2t = ss2t_bad, est_dms = dms_bad)    
        
    def test_rejects_unphysical_LE(self):     
        num_numu, num_nue = 100, 1
        L_test, E_test = (-1, -1)
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, E = E_test)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, L = L_test)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, E = E_test, L = L_test)
        
    def test_rejects_unphysical_stdLE(self):     
        num_numu, num_nue = 100, 1
        stdL_test, stdE_test = (-1, -1)
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, std_E = stdE_test)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, std_L = stdL_test)    
        self.assertRaises(ValueError, module.fit_model, num_numu, num_nue, std_E = stdE_test, std_L = stdL_test)  
        
class TestBinnedOscillation_model(TestCase):    
    
    def test_produces_a_model(self):    
        bins = [0, 1, 2]
        nue = [1, 1]
        test_model = module.binned_oscillation_model(10, nue, bins)      
        self.assertTrue(isinstance(test_model, pm.model.Model))    
        
    def test_rejects_unresonable_inputs(self):        
        num_numu = 10
        num_nue = [1, 11, 0]
        bins = [0, 2]
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins)
        
    def test_rejects_unphysical_mixing_params(self):     
        num_numu = 100
        ss2t_bad, ss2t_neg = 22, -1         
        dms_bad = -0.01
        bins = [0, 1, 2]
        num_nue = [1, 1]
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, est_ss2t = ss2t_bad)    
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, est_ss2t = ss2t_neg)    
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, est_dms = dms_bad)
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, est_ss2t = ss2t_bad, est_dms = dms_bad)    
        
    def test_rejects_unphysical_L(self):     
        num_numu=100
        L_test, E_test = (-1, -1)
        bins = [0, 1, 2]       
        num_nue = [1, 1]
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, L = L_test)    
        
    def test_rejects_unphysical_stdL(self):     
        num_numu=10
        stdL_test, stdE_test = (-1, -1) 
        bins= [0, 1, 2]
        num_nue = [1, 1]
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins, std_L = stdL_test)    
    
    def test_nue_shape(self):
        bins = [0, 1, 2]
        num_nue = [1, 100]
        num_numu = [10, 19]
        self.assertRaises(ValueError, module.binned_oscillation_model, num_numu, num_nue, bins)


class TestBinnedInference(TestCase):    
    #Apparently we might not need these - not deleting for now just in case, but probably will in a later update
    ss2t = 0.1 #unit-less
    dms = 1 #eV^2
    E = 1 #GeV
    L = 0.5 #km
    num_numu = 600000
    num_nue = num_numu*ss2t*np.sin(1.27*dms*L/E)**2
   
    def test_fit_model_guess_dict(self):
        bad_guess = {'el': 0}
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1]
   
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue, bins, initial_guess = bad_guess)
        
    def test_fit_model_guess_values(self):
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1]
   
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue,bins, initial_guess ={'L': -1.0})
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue,bins, initial_guess ={'sin^2_2theta': -1.0})
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue,bins, initial_guess ={'sin^2_2theta': 1.1})
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue,bins, initial_guess ={'delta_m^2': -1.0})
        self.assertRaises(ValueError, module.binned_fit_model, TestBinnedInference.num_numu, TestBinnedInference.num_nue,bins, initial_guess ={'rate': -1.0})                    
            
    def test_rejects_unresonable_inputs(self):        
        num_numu, num_nue = (10, 22)         
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1] 
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue, bins)
        
    def test_rejects_unphysical_mixing_params(self):     
        num_numu, num_nue = 100, 1
        ss2t_bad, ss2t_neg = 22, -1         
        dms_bad = -0.01
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1]
   
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, est_ss2t = ss2t_bad)    
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, est_ss2t = ss2t_neg)    
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, est_dms = dms_bad)
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, est_ss2t = ss2t_bad, est_dms = dms_bad)    
        
    def test_rejects_unphysical_L(self):     
        num_numu=100
        L_test, E_test = (-1, -1) 
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1]
   
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, L = L_test)    
        
    def test_rejects_unphysical_stdL(self):     
        num_numu=100
        stdL_test, stdE_test = (-1, -1) 
        bins = [0, 1, 2, 3]
        num_nue = [1, 1, 1]
   
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue,bins, std_L = stdL_test)    

    def test_valid_binning(self):
        bad_bins = [0, 1, 2, 4, 3]
        num_nue = [1, 1, 1]   
        num_numu= 100
        self.assertRaises(ValueError, module.binned_fit_model, num_numu, num_nue, bad_bins)    
