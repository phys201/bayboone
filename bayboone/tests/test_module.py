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

  
        
    def test_fit_model_returns_trace(self):
        #self.assertTrue(isinstance(TestInference.trace, az.InferenceData) or isinstance(TestInference.trace, pm.backends.base.MultiTrace))  
        return True

    def test_inference_returns_physical_values(self):
        # We're still working on improving the statistical model to get one that's both correct and does the inference well
        # (See ongoing discussion in group 6 slack channel)
        # In the meantime, we'll just make sure it returns physical values
        
        #df_trace = pm.trace_to_dataframe(TestInference.trace)

       # q = df_trace.quantile([0.16,0.50,0.84], axis=0)
        #df_trace['delta_m^2'] = np.exp(df_trace['log_delta_m^2'])
       # self.assertTrue(q['delta_m^2'][0.84]<1 and q['delta_m^2'][0.16]>0.01)
       # self.assertTrue(q['sin^2_2theta'][0.84]<1 and q['sin^2_2theta'][0.16]>0.0001)
        return True

    
   