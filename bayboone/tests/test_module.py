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
        
class TestInference(TestCase):    
    
    ss2t = 0.1 #unit-less
    dms = 1 #eV^2
    E = 1 #GeV
    L = 0.5 #km
    num_numu = 600000
    num_nue = num_numu*ss2t*np.sin(dms*L/E)**2
    trace = module.fit_model(num_numu, num_nue, 1000)
    
    def test_fit_model_returns_trace(self):
        self.assertTrue(isinstance(TestInference.trace, az.InferenceData) or isinstance(TestInference.trace, pm.backends.base.MultiTrace))    

    def test_inference_returns_physical_values(self):
        # We're still working on improving the statistical model to get one that's both correct and does the inference well
        # (See ongoing discussion in group 6 slack channel)
        # In the meantime, we'll just make sure it returns physical values
        
        df_trace = pm.trace_to_dataframe(TestInference.trace)

        q = df_trace.quantile([0.16,0.50,0.84], axis=0)
        #df_trace['delta_m^2'] = np.exp(df_trace['log_delta_m^2'])
        self.assertTrue(q['delta_m^2'][0.84]<1 and q['delta_m^2'][0.16]>0.01)
        self.assertTrue(q['sin^2_2theta'][0.84]<1 and q['sin^2_2theta'][0.16]>0.0001)

    def test_prints_fitvals(self):
        self.assertTrue(module.print_fit_vals(TestInference.trace))
    
   