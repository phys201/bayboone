import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
# Adapted from week 07 notebook

def oscillation_model(num_neutrinos, num_nue):
    '''
    Creates a statistical model for predicting the oscillation parameters from microboone-like values
    Inputs:
    num_neutrinos: 
        The number of muon neutrinos shot at the detector
    num_nue:
        The number of electron neutrinos detected
    Returns:
    osc_model: pymc3 model
        the statistical model for our neutrino oscillations in the 3+1 model
    '''
    #Check that the data is reasonable
    if (num_neutrinos < num_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")

    osc_model = pm.Model()
    with osc_model:
    
        # Priors for unknown model parameters, covering the sensitivity region of modern experiments
        ss2t = pm.Uniform('sin^2_2theta', 0.0001, 1)
        #dms = pm.Uniform('delta_m^2', 0.01, 11.0) #units of ev^2
        log_dms = pm.Uniform('log_delta_m^2', -4.605, 2.3)
        # We don't know the exact energy or production point of each neutrino, so we draw from a truncated gaussian 
        # (enforcing positive distance travelled and energy)
        
        L = pm.TruncatedNormal('L', mu = 0.500, sigma = 0.05, lower = 0) #units of km
        E = pm.TruncatedNormal('E', mu = 1.0, sigma = 0.05, lower = 0) #units of GeV

        # In the large n limit, because the number of oscillations is low, we use a Poisson approximation
        #rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(dms*(1.27*L)/E))**2)
        rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(np.exp(log_dms)*(1.27*L)/E))**2)
        
        #Likelihood of observations
        measurements = pm.Poisson('nue_flux', mu = rate, observed = num_nue)
        
    return osc_model

def fit_model(num_neutrinos, num_nue, num_draws = 1000):
    '''
    Fits a given model to data provided using MCMC sampling
    
    Inputs:
    num_neutrinos: float
        the number of muon neutrinos produced 
    num_nue: number of electron neutrinos detected
        e- neutrinos detected 
    num_draws: int
        number of samples to draw
        
    Returns:
    trace: arviz InferenceData object
        results of the mcmc sampling procedure
        
        '''
    uncertainty = np.sqrt(num_nue)
    osc_model = oscillation_model(num_neutrinos, num_nue)
    
    with osc_model:
        trace = pm.sample(num_draws)
        az.plot_trace(trace)
        
    return trace
                         
def print_fit_vals(trace):
    
    '''
    literally just prints the best fit values and their uncertainties nicely
    
    Inputs:
    trace: arviz InferenceData object
        Holds the results from the mcmc sampling procedure
    
    '''
    df_trace = pm.trace_to_dataframe(trace)
    df_trace['delta_m^2'] = np.exp(df_trace['log_delta_m^2'])
    q = df_trace.quantile([0.16,0.50,0.84], axis=0)
    print("delta_m^2 = {:.2f} + {:.2f} - {:.2f}".format(q['delta_m^2'][0.50], 
                                            q['delta_m^2'][0.84]-q['delta_m^2'][0.50],
                                            q['delta_m^2'][0.50]-q['delta_m^2'][0.16]))
    print("sin^2_2theta = {:.1f} + {:.1f} - {:.1f}".format(q['sin^2_2theta'][0.50], 
                                            q['sin^2_2theta'][0.84]-q['sin^2_2theta'][0.50],
                                            q['sin^2_2theta'][0.50]-q['sin^2_2theta'][0.16]))
    
    return True
    