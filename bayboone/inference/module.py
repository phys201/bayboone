import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from numpy.linalg import inv, eigh, det
# Adapted from week 07 notebook

def oscillation_model(num_neutrinos, num_nue, est_ss2t = 0.5, est_dms = 0.8):
    '''
    Creates a statistical model for predicting the oscillation parameters from microboone-like values
    
    Inputs
        num_neutrinos:
            The number of muon neutrinos shot at the detector
        num_nue:
            The number of electron neutrinos detected
        est_ss2t: float between 0 and 1
            estimated ss2t from previous experiments, for use in the prior
        est_dms: float above 0
            estimated dms from previous experiments, also for use in prior
            
    Returns
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

        measurements = pm.Poisson('nue_flux', mu = rate, observed = num_nue)

    return osc_model

def fit_model(num_neutrinos, num_nue, num_draws = 10000):
    '''
    Fits a given model to data provided using MCMC sampling

    Inputs
        num_neutrinos: float
            the number of muon neutrinos produced
        num_nue: number of electron neutrinos detected
            e- neutrinos detected
        num_draws: int
            number of samples to draw
        model: int (3 or 4)
            The model of either 3 or 4 neutrinos

    Returns
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
    Prints the best fit values and their uncertainties nicely

    Inputs
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


uncertainty = 0.1
def run_fit(initial_guess, model):
    """
    Performs fit of a given model
    
    Inputs
        initial_guess: dictionary containing initial guess values
        
    Returns
        best_fit: dictionary containing best fit values, covariance matrix, forward function and inital guess values
    """
    best_fit, scipy_output = pm.find_MAP(model=model, start = initial_guess, return_raw=True)
    covariance_matrix = np.flip(scipy_output.hess_inv.todense()/uncertainty)

    best_fit['covariance matrix'] = covariance_matrix
    best_fit['forward function'] = model.fn(model['rate'])
    best_fit['initial guess'] = initial_guess
    return best_fit

def calc_residuals(data, best_fit):
    """
    Calculates the residuals of a given fit 
    
    Inputs
        best_fit: dictionary containing best fit values (must contain 'prediction')
    
    Returns
        ndarray of residuals 
    """
    return data.N_nue - best_fit['rate'].flatten()*data.N_numu

def calc_chi_squared(data,best_fit):
    """
    Calculates chi squared for a given fit
    
    Inputs
        best_fit: dictionary containing best fit values
    
    Returns
        chi squared value (float)
    """
    residuals = calc_residuals(data,best_fit)/uncertainty
    return np.sum(residuals**2)

def global_log_likelihood(data, best_fit):
    """
    Calculates the global log likelihood for a given fit
    
    Inputs
        best_fit: dictionary containing best fit values
    
    Returns 
        global log likelihood (float)
    """
    n_parameters = len(best_fit['initial guess'])
    chi_squared = calc_chi_squared(data, best_fit)
    max_likelihood =  np.exp(-chi_squared/2) / (2 * np.pi * uncertainty**2) ** (1/2)
    curvature = np.sqrt(det(best_fit['covariance matrix'])) * (2 * np.pi) **  (n_parameters/2)

    #n_peaks = int((n_parameters - 1) / 2)
    #prior_L = 1 / np.ptp(d['L'])
    #prior_peak_center = 1 / np.ptp(data11_1['f']) ** n_peaks
    log_prior = 0 #np.log(prior_background * prior_peak_height * prior_peak_center)

    return np.log(max_likelihood) + np.log(curvature) + log_prior

def compute_odds(data,best_fit, previous_fit):
    """
    Calculates the odds ratio between two fits (current/previous)
    
    Inputs
        best_fit: dictionary containing best fit values of current model
        previous_fit: dictionary containing best fit values of the previous model 
    
    Returns
        nothing; prints global likelihood of current model and odds ratio
    """
    previous_log_like = global_log_likelihood(data,previous_fit)
    current_log_like = global_log_likelihood(data,best_fit)
    odds = np.exp(current_log_like - previous_log_like)
    print("\nGlobal likelihood:\n{}".format(np.exp(current_log_like)))
    print("A model that includes a fourth neutrino is favored by a factor of {:.0f} compared to one that doesn't.".format(odds))

