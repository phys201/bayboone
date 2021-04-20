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
        the statistical model for our neutrino oscillations
    '''
    #Check that the data is reasonable
    if (num_neutrinos < num_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")

    osc_model = pm.Model()
    with osc_model:
    
        # Priors for unknown model parameters
        ss2t = pm.Uniform('sin^2_2theta', 0, 1)
        dms = pm.Uniform('delta_m^2', 0, 1.0) #units of ev^2
        # We don't know the exact energy or production point of each neutrino, so we draw from a truncated  gaussian (enforcing positive distance travelled and energy)
        #L_over_E = pm.Normal('L_over_E', mu = 0.5, sigma = 0.1 ) #units of km/Gev
        L = pm.TruncatedNormal('L', mu = 0.500, sigma = 0.05, lower = 0) #units of km
        E = pm.TruncatedNormal('E', mu = 1.0, sigma = 0.05, lower = 0) #units of GeV

        # Expected value from theory 
        #P = pm.Deterministic('prediction', ss2t*(np.sin(dms*(1.27*L)/E))**2)
        # Likelihood of observations
        # measurements = pm.Binomial('nue_Flux', n=num_neutrinos, p=P, observed=num_nue)
        
        # In the large n limit, because the number of oscillations is low, we use a Poisson approximation
        # Rate parameter calculated form theory
        rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(dms*(1.27*L)/E))**2)
        
        #Likelihood of observations
        measurements = pm.Poisson('nue_flux', mu = rate, observed = num_nue)
        
    return osc_model

def fit_model(data):
    '''Fits a given model to data provided
    Inputs:
    data: two floats
        the number of muon neutrinos produced (data[0]) and e- neutrinos detected (data[1])'''
    
    num_neutrinos = data[0]
    num_nue = data[1]
    uncertainty = np.sqrt(num_nue)
    osc_model = oscillation_model(num_neutrinos, num_nue)
    
    with osc_model:
        trace = pm.sample(10000)
        az.plot_trace(trace)
        
    return trace

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
    uncertainty = 0.003
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
    #values[1]= bf['dmsL_over_E']
    #rows[1] = 'dmsL_over_E'
    uncertainty[1] = np.sqrt(cov[1][1])
    #We don't report values for our nusiance parameters in this function
    
    vals = { 'value': values}
    fit_values = pd.DataFrame(vals, index = rows)
    fit_values['uncertainty'] = uncertainty
    
    return fit_values
    
def do_inference(data):
    'an easy function to call that infers values of ss2t, dms, from a given data set'
    
    import matplotlib.pyplot as plt
    best_fit, cov = fit_model(data)
    data.plot(x='E', y='L', kind='scatter', yerr=0.5)
    x = np.linspace(min(data['E']), max(data['E']), len(data['E']))

    plt.plot(x, best_fit['prediction'], '-k', color='red')
    print_fit_vals(best_fit, cov)
    
    return