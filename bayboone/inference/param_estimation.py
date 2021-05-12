import numpy as np
import pymc3 as pm 
from collections.abc import Iterable

UC = 1.27 #Unit conversion contsant in the oscillation probability

def oscillation_model(num_neutrinos, num_nue, est_ss2t = 0.5, est_dms = 0.8, L = 0.54, std_L = 0.015, E = 1.0, std_E = 0.15):
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
    L: float above 1
        mean distance travelled by a neutrino in km
    std_L: float above 1
        standard deviation in L
    E: float above 1
        mean neutrino energy in GeV
    std_E:
        standard deviation in L
        
    Returns:
    osc_model: pymc3 model
        the statistical model for neutrino oscillations in the 3+1 model

    '''
    #Check that the data is reasonable
    if (num_neutrinos < num_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")
    
    #Check physical values of est_ss2t, est_dms are provided
    if (est_ss2t>1 or est_ss2t<0 or est_dms<0):
        raise ValueError("Error: estimated sin^2(2theta) must be between 0 and 1, and estimated delta m^2 must be greater than zero")
        
    #Check L, E > 0
    if (L<0 or E<0):
        raise ValueError("Error: L and E must be greater than zero")
        
    #Check std_L, std_E > 0
    if (std_L<0 or std_E<0):
        raise ValueError("Error: standard deviations of L and E must be greater than zero")

    #Create the model
    osc_model = pm.Model()
    with osc_model:
        
        # We don't know the exact energy or production point of each neutrino, so we draw from a truncated gaussian 
        # (enforcing positive distance travelled and energy)   
        L = pm.TruncatedNormal('L', mu = L, sigma = std_L, lower = 0, upper = 0.6) #units of km
        E = pm.TruncatedNormal('E', mu = E, sigma = std_E, lower = 0) #units of GeV
        
        # Priors for unknown model parameters, centered on a prior estimate of ss2t, dms
        ss2t = pm.TruncatedNormal('sin^2_2theta', mu = est_ss2t, sigma = 0.5, lower = 0, upper = 1 ) #pm.Uniform('sin^2_2theta', 0.0001, 1)
        dms = pm.TruncatedNormal('delta_m^2', mu = est_dms, sigma = 0.5, lower = 0, upper = E*np.pi/(1.27*L))
        #"upper" limits the inferred point to only the first of the delta m^2 values that would fit the function, eliminating periodicity.
        
        # In the large n limit, because the number of oscillations is low, we use a Poisson approximation
        rate = pm.Deterministic('rate', num_neutrinos*ss2t*(np.sin(dms*(UC*L)/E))**2)
        
        #Likelihood of observations
        measurements = pm.Poisson('nue_flux', mu = rate, observed = num_nue)

    return osc_model

def fit_model(num_neutrinos, num_nue, num_draws = 1000, initial_guess = None, 
              est_ss2t = 0.5, est_dms = 0.8, L = 0.54, std_L = 0.015, E = 1.0, std_E = 0.15):
    '''
    Fits a given model to data provided using MCMC sampling
    
    Inputs:
    num_draws: int
        number of samples to draw
    initial_guess: dictionary 
        start guesses of model parameters for inference
        
    Inputs passed to osc_model:
    num_neutrinos: int REQUIRED
        The number of muon neutrinos shot at the detector
    num_nue: int REQUIRED
        The number of electron neutrinos detected
    est_ss2t: float between 0 and 1
        estimated ss2t from previous experiments, for use in the prior
    est_dms: float above 0
        estimated dms from previous experiments, also for use in prior
    L: float above 1
        mean distance travelled by a neutrino in km
    std_L: float above 1
        standard deviation in L
    E: float above 1
        mean neutrino energy in GeV
    std_E:
        standard deviation in L
        
    Returns:
    trace: arviz InferenceData object
        results of the mcmc sampling procedure
        
        '''
    #Check that the data is reasonable
    if (num_neutrinos < num_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")
    
    #Check model inputs (as in osc_model)
    if (est_ss2t>1 or est_ss2t<0 or est_dms<0):
        raise ValueError("Error: estimated sin^2(2theta) must be between 0 and 1, and estimated delta m^2 must be greater than zero")
    if (L<0 or E<0):
        raise ValueError("Error: L and E must be greater than zero")
    if (std_L<0 or std_E<0):
        raise ValueError("Error: standard deviations of L and E must be greater than zero")
        
    #Check any initial guess keys represent actual model parameters
    allowed_keys = {'E', 'L', 'sin^2_2theta', 'delta_m^2', 'rate'}
    if (initial_guess !=None):
        for key in initial_guess:
            if key in allowed_keys:
                continue
            else: 
                raise ValueError("Incorrect Key: allowed keys for initial guess are: 'E', 'L', 'sin^2_2theta', 'delta_m^2', 'rate'")
        #Check initial guesses, if provided, are physical
        if 'E' in initial_guess:
            if initial_guess.get('E')<0: raise ValueError("Error: Guess for E cannot be less than zero")
        if 'L' in initial_guess:
            if initial_guess.get('L')<0: raise ValueError("Error: Guess for L cannot be less than zero")
        if 'sin^2_2theta' in initial_guess:
            if (initial_guess.get('sin^2_2theta')<0 or initial_guess.get('sin^2_2theta')>1): raise ValueError("Error: Guess for sin^2(2theta) must be between 0 and 1")
        if 'delta_m^2' in initial_guess:
            if initial_guess.get('delta_m^2')<0: raise ValueError("Error: Guess for delta_m^2 cannot be less than zero")
        if 'rate' in initial_guess:
            if initial_guess.get('rate')<0: raise ValueError("Error: Guess for rate cannot be less than zero")

    
    uncertainty = np.sqrt(num_nue)
    osc_model = oscillation_model(num_neutrinos, num_nue, est_ss2t, est_dms, L , std_L, E, std_E)
    
    with osc_model:
        trace = pm.sample(num_draws, start=initial_guess)
        
    return trace

def binned_oscillation_model(num_neutrinos, num_nue, energy_bins, initial_guess = None, est_ss2t = 0.5, est_dms = 0.8, L = 0.54, std_L = 0.015):
    '''
    Creates a statistical model for predicting the oscillation parameters from microboone-like values
    Inputs:
    num_neutrinos: float or array of floats
        The number of muon neutrinos shot at the detector in each energy bin
        To see a single rate curve, use a float. To see an island fit for each chain, use and array of floats.
    num_nue: array of floats
        The number of electron neutrinos detected in each energy bin
    energy_bins: array of floats, size at least 2
        bins edges for the energies we've binned data into
    est_ss2t: float between 0 and 1
        estimated ss2t from previous experiments, for use in the prior
    est_dms: float above 0
        estimated dms from previous experiments, also for use in prior
    Returns:
    osc_model: pymc3 model
        the statistical model for our neutrino oscillations in the 3+1 model with multiple energy bins
    '''
    
    #Check fewer nues than numus
    all_nue = np.sum(num_nue)
    all_numu = np.sum(num_neutrinos)
    if (all_numu<all_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")
    
    if(isinstance(num_neutrinos, Iterable)):
        for k in enumerate(num_neutrinos):
            if (k[1]<num_nue[k[0]]):
                raise ValueError("Error: number of initial neutrinos in a bin cannot be less than number of nues observed in that bin")

    #Check num_nue has correct shape
    if (len(num_nue) != (len(energy_bins)-1)):
        raise ValueError("Error: Please specify one value for number of nues observed per energy bin")   
        
    # Check energy bins are in order
    energy_bins = np.array(energy_bins)
    if((energy_bins != sorted(energy_bins)).all()):
        raise ValueError("Error: Bin edges must be in order low-high")

    #Check model inputs (as in osc_model)
    if (est_ss2t>1 or est_ss2t<0 or est_dms<0):
        raise ValueError("Error: estimated sin^2(2theta) must be between 0 and 1, and estimated delta m^2 must be greater than zero")
    if (L<0 or std_L<0):
        raise ValueError("Error: L and std_L must be greater than zero")
        
    #Check any initial guess keys represent actual model parameters
    allowed_keys = {'L', 'sin^2_2theta', 'delta_m^2', 'rate'}
    if(initial_guess != None):
        for key in initial_guess:
            if key in allowed_keys:
                continue
            else: 
                raise ValueError("Incorrect Key: allowed keys for initial guess are: 'L', 'sin^2_2theta', 'delta_m^2', 'rate'")

   
    energies = np.zeros(energy_bins.size-1)
    sigmas = np.zeros(energy_bins.size-1)
    for i in range(energies.size):
        energies[i] = energy_bins[i] + (energy_bins[i+1]-energy_bins[i])/2
        sigmas[i] = (energy_bins[i+1]-energy_bins[i]) # Most energy reconstructions aim for the standard deviation to be about the bin width. 
    
    energies_high = np.array(energy_bins[1:])
    energies_low = np.array(energy_bins[:-1])

    osc_model = pm.Model()
    with osc_model:

        # We don't know the exact production point of each neutrino, so we draw from a truncated gaussian (enforcing positive distance travelled)   
        L = pm.TruncatedNormal('L', mu = 0.540, sigma = 0.015, lower = 0.02, upper = 0.6) #units of km

        #If this works, we'll have one E distribution for each energy bin
        #E = energies #
        E = pm.TruncatedNormal('E', mu = energies, sigma = sigmas, lower = energies_low, shape=energies.shape[0]) #units of GeV

        # Priors for unknown model parameters, centered on a prior estimate of ss2t, dms
        # Est_ss2t, est_dms defined in previous cell, will be input parameters in our function
        ss2t = pm.TruncatedNormal('sin^2_2theta', mu = est_ss2t, sigma = 0.1, lower = 0, upper = 1 ) #pm.Uniform('sin^2_2theta', 0.0001, 1)
        dms = pm.TruncatedNormal('delta_m^2', mu = est_dms, sigma = 0.1, lower = 0)

        # In the large n limit, because the number of oscillations is low, we use a Poisson approximation
        rate = pm.Deterministic('rate', ss2t*(np.sin(dms*(UC*L)/E))**2)

        #Likelihood of observations
        measurements = pm.Poisson('nue_flux', mu = rate*num_neutrinos, observed = num_nue)
        
    return osc_model

def binned_fit_model(num_neutrinos, num_nue, energy_bins, num_draws = 1000, initial_guess=None,
              est_ss2t = 0.5, est_dms = 0.8, L = 0.54, std_L = 0.015):
    '''
    Fits a given model to data provided using MCMC sampling
    
    Inputs:
    num_neutrinos: float
        the number of muon neutrinos produced 
    num_nue: number of electron neutrinos detected
        e- neutrinos detected 
    energy_bins: array of floats
        bins edges for the energies we've biined data into
    num_draws: int
        number of samples to draw
        
    Returns:
    trace: arviz InferenceData object
        results of the mcmc sampling procedure
        
        '''
    # Check energy bins are in order
    energy_bins = np.array(energy_bins)
    sorted_bins = np.sort(energy_bins)
    if((energy_bins != sorted_bins).any()):
        raise ValueError("Error: Bin edges must be in order low-high")
        
    #Check fewer nues than numus
    all_nue = np.sum(num_nue)
    all_numu = np.sum(num_neutrinos)
    if (all_numu<all_nue):
        raise ValueError("Error: number of initial neutrinos cannot be less than number of nues observed")
    
    if(isinstance(num_neutrinos, Iterable)):
        for k in enumerate(num_neutrinos):
            if (k[1]<num_nue[k[0]]):
                raise ValueError("Error: number of initial neutrinos in a bin cannot be less than number of nues observed in that bin")
    
    #Check model inputs (as in osc_model)
    if (est_ss2t>1 or est_ss2t<0 or est_dms<0):
        raise ValueError("Error: estimated sin^2(2theta) must be between 0 and 1, and estimated delta m^2 must be greater than zero")
    if (L<0 or std_L<0):
        raise ValueError("Error: L and std_L must be greater than zero")
        
    #Check any initial guess keys represent actual model parameters
    allowed_keys = {'L', 'sin^2_2theta', 'delta_m^2', 'rate'}
    if(initial_guess != None):
        for key in initial_guess:
            if key in allowed_keys:
                continue
            else: 
                raise ValueError("Incorrect Key: allowed keys for initial guess are: 'L', 'sin^2_2theta', 'delta_m^2', 'rate'")

        #Check initial guesses, if provided, are physical
        if 'L' in initial_guess:
            if initial_guess.get('L')<0: raise ValueError("Error: Guess for L cannot be less than zero")
        if 'sin^2_2theta' in initial_guess:
            if (initial_guess.get('sin^2_2theta')<0 or initial_guess.get('sin^2_2theta')>1): raise ValueError("Error: Guess for sin^2(2theta) must be between 0 and 1")
        if 'delta_m^2' in initial_guess:
            if initial_guess.get('delta_m^2')<0: raise ValueError("Error: Guess for delta_m^2 cannot be less than zero")
        if 'rate' in initial_guess:
            if initial_guess.get('rate')<0: raise ValueError("Error: Guess for rate cannot be less than zero")

        
    osc_model = binned_oscillation_model(num_neutrinos, num_nue, energy_bins)
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
    q = df_trace.quantile([0.16,0.50,0.84], axis=0)
    print("delta_m^2 = {:.2f} + {:.2f} - {:.2f}".format(q['delta_m^2'][0.50], 
                                            q['delta_m^2'][0.84]-q['delta_m^2'][0.50],
                                            q['delta_m^2'][0.50]-q['delta_m^2'][0.16]))
    print("sin^2_2theta = {:.1f} + {:.1f} - {:.1f}".format(q['sin^2_2theta'][0.50],
                                            q['sin^2_2theta'][0.84]-q['sin^2_2theta'][0.50],
                                            q['sin^2_2theta'][0.50]-q['sin^2_2theta'][0.16]))