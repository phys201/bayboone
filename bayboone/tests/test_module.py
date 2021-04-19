from unittest import TestCase

#Dependencies of our module
import numpy as np
import pandas as pd
import pymc3 as pm

from bayboone.inference import module

class TestOscillation_model(TestCase):    
    
    def test_produces_a_model(self):        
        test_model = module.oscillation_model(6000, 1)      
        self.assertTrue(isinstance(test_model, pm.model.Model))    
        
    def test_accepts_resonable_inputs(self):        
        num_numu, num_nue = (10, 22)        
        #test_model = module.oscillation_model(num_numu, num_nue)       
        self.assertRaises(ValueError, module.oscillation_model, num_numu, num_nue)    
        
class TestInference(TestCase):
    
    def test_fit_model(self):
        ss2t = 0.1 #unit-less
        dms = 1 #eV^2
        E = 1 #GeV
        L = 0.5 #km
        num_numu = 600000
        num_nue = num_numu*ss2t*np.sin(dms*L/E)**2
        
        best_fit, cov = module.fit_model([num_numu, num_nue])
        
        self.assertAlmostEqual(ss2t, best_fit['sin^2_2theta'])
        self.assertAlmostEqual(dms, best_fit['delta_m^2'])
        
    
        