import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
# Adapted from week 07 notebook

def oscillation_model(fake_data):
    uncertainty = 1.0 
    #This is chosen arbitrarily - we'll re-evaluate to get a better value when we work on making our model reflective of reality

    # reshape data so it behaves when pymc3 tests multiple parameter values at once
    L = fake_data['L'].values[:, np.newaxis]
    E = fake_data['E'].values[:, np.newaxis]
    num_neutrinos = fake_data['N_numu']
    num_nue = fake_data['N_nue']

    # The following two lines set up the model, which is a Python object.  
    # "with peaks_model" is called a context manager: It provides a convenient way to set up the object. 
    osc_model = pm.Model()
    with osc_model:
    
        # Priors for unknown model parameters
        ss2t = pm.Uniform('sin^2_2theta', 0, 1)
        dms = pm.Uniform('delta_m^2', 0, 0.01)  
        
        # Expected value from theory 
        P = pm.Deterministic('prediction', ss2t*(np.sin(dms*(1.27*L)/E))**2)
        
        # Likelihood of observations
        # Oscillation from numu to nue is like a weighted coin toss, so we use the binomial distribution
        measurements = pm.Binomial('nue_Flux', n=num_neutrinos, p=P, observed=num_nue)
        
    return osc_model

def fit_model(data, initial_guess = {'sin^2_2theta':0.1, 'delta_m^2':0.001}):
    
    uncertainty = 0.3
    osc_model = oscillation_model(data)
    best_fit, scipy_output = pm.find_MAP(model=osc_model, start = initial_guess, return_raw=True)    
    covariance_matrix = np.flip(scipy_output.hess_inv.todense()/uncertainty)
    
    return best_fit, covariance_matrix

def chisq(fake_data, prediction):
    '''
    Finds chi-squared for a given model
    
    Parameters
    --------
    prediction: numpy array
        the prediction from best-fit of for a given model 
        (the best fit means this gives the minimum chi squared)
    x_vals: numpy array
        the y values for our data
    
    Returns
    --------
    float:
        the chi-squared
    '''   
    uncertainty = 0.3
    observed = fake_data['N_nue']
    res_squared = np.power(prediction-observed, 2)
    chisq = np.sum(res_squared/uncertainty**2)
    
    return chisq
                         
def print_fit_vals(bf, cov):
    
    '''
    literally just prints the best fit values and their uncertainties nicely
    
    Inputs:
    bf: best fit from the model
    cov: covariance matrix from the model
    '''
    points = len(cov[0])
    values = np.zeros(points)
    rows = np.full(points, '00000000000000')
    uncertainty = np.zeros(points)
    
    values[0] = bf['sin^2_2theta']
    rows[0] = 'sin^2_2theta'
    uncertainty[0] = np.sqrt(cov[0][0])

    values[1] = bf['delta_m^2']
    rows[1] = 'delta_m^2'
    uncertainty[1] = np.sqrt(cov[1][1])
    
    vals = { 'value': values}
    fit_values = pd.DataFrame(vals, index = rows)
    fit_values['uncertainty'] = uncertainty
    
    print(fit_values)

    return fit_values
    
def do_inference(data):
    'an easy function to call that infers values of ss2t, dms, from a given data set, for use in the tutorial'
    
    import matplotlib.pyplot as plt
    best_fit, cov = fit_model(data)
    data.plot(x='E', y='L', kind='scatter', yerr=0.5)
    x = np.linspace(min(data['E']), max(data['E']), len(data['E']))

    plt.plot(x, best_fit['prediction'], '-k', color='red')
    print_fit_vals(best_fit, cov)
    
    return