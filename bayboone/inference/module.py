import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
# Adapted from week 07 notebook

def oscillation_model(num_neutrinos, num_nue, est_ss2t = 0.5, est_dms = 0.8):
    '''
    Creates a statistical model for predicting the oscillation parameters from microboone-like values
    Inputs:
    num_neutrinos: 
        The number of muon neutrinos shot at the detector
    num_nue:
        The number of electron neutrinos detected
    est_ss2t: float between 0 and 1
        estimated ss2t from previous experiments, for use in the prior
    est_dms: float above 0
        estimated dms from previous experiments, also for use in prior
    Returns:
    osc_model: pymc3 model
        the statistical model for our neutrino oscillations in the 3+1 model
    '''
    #Check that the data is reasonable
    if (num_neutrinos < num_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")

    osc_model = pm.Model()
    with osc_model:
        
        # We don't know the exact energy or production point of each neutrino, so we draw from a truncated gaussian 
        # (enforcing positive distance travelled and energy)   
        L = pm.TruncatedNormal('L', mu = 0.500, sigma = 0.15, lower = 0, upper = 0.6) #units of km
        E = pm.TruncatedNormal('E', mu = 1.0, sigma = 0.15, lower = 0) #units of GeV
        
        # Priors for unknown model parameters, centered on a prior estimate of ss2t, dms
        ss2t = pm.TruncatedNormal('sin^2_2theta', mu = est_ss2t, sigma = 0.1, lower = 0, upper = 1 ) #pm.Uniform('sin^2_2theta', 0.0001, 1)
        dms = pm.TruncatedNormal('delta_m^2', mu = est_dms, sigma = 0.1, lower = 0, upper = E*np.pi/(1.27*L))
        #This limits the inferred point to only the first of the delta m^2 values that would fit the function, eliminating periodicity.
        
        #pm.Uniform('delta_m^2', 0.01, 11.0) #units of ev^2
        #log_dms = pm.Uniform('log_delta_m^2', -4.605, 2.3)
        

        # In the large n limit, because the number of oscillations is low, we use a Poisson approximation
        rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(dms*(1.27*L)/E))**2)
        #rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(np.exp(log_dms)*(1.27*L)/E))**2)
        
        #Likelihood of observations
        measurements = pm.Poisson('nue_flux', mu = rate, observed = num_nue)
        
    return osc_model

def fit_model(num_neutrinos, num_nue, num_draws = 10000):
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
    initial_guess = {'L': 0.5, 'E': 0.5, 'sin^2_2theta': 0.3, 'delta_m^2': 1}
    
    with osc_model:
        trace = pm.sample(num_draws, start=initial_guess)
        
    return trace
                         
def print_fit_vals(trace):
    
    '''
    literally just prints the best fit values and their uncertainties nicely
    
    Inputs:
    trace: arviz InferenceData object
        Holds the results from the mcmc sampling procedure
    
    '''
    df_trace = pm.trace_to_dataframe(trace)
    #df_trace['delta_m^2'] = np.exp(df_trace['log_delta_m^2'])
    q = df_trace.quantile([0.16,0.50,0.84], axis=0)
    print("delta_m^2 = {:.2f} + {:.2f} - {:.2f}".format(q['delta_m^2'][0.50], 
                                            q['delta_m^2'][0.84]-q['delta_m^2'][0.50],
                                            q['delta_m^2'][0.50]-q['delta_m^2'][0.16]))
    print("sin^2_2theta = {:.1f} + {:.1f} - {:.1f}".format(q['sin^2_2theta'][0.50], 
                                            q['sin^2_2theta'][0.84]-q['sin^2_2theta'][0.50],
                                            q['sin^2_2theta'][0.50]-q['sin^2_2theta'][0.16]))
    
    return True
    