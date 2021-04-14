import os
import pandas as pd
import numpy as np


def get_data_file_path(filename, data_dir='data'):
    # __file__ is the location of the source file currently in use (so
    # in this case io.py). We can use it as base path to construct
    # other paths from that should end up correct on other machines or
    # when the package is installed
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    # If you need to go up another directory (for example if you have
    # this function in your tests directory and your data is in the
    # package directory one level up) you can use
    # up_dir = os.path.split(start_dir)[0]
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)


def load_data(data_file):
    return pd.read_csv(data_file, sep=' ')

def generative_model(n_points=100, stheta=0.5, dm=10e-5):
    """
    Simulates data based on a generative model of the 4-neutrino
    oscillation probability. 
    
    inputs:
        n_points: int
            The number of data points to generate
        stheta: float between 0 and 1
            The oscillation parameter sin^2(2*theta_14)
        dm: float
            The oscillation parameter delta(m_14)^2
            
    Returns:
        data: pandas dataframe
            A dataframe holding the simulated events with columns 
            for the number of electron neutrinos (N_nue), the 
            number of muon neutrinos (N_numu), the beam 
            length (L), the number of initial muon neutrinos 
            (N_numu_initial), and the beam energy (E)
    """
    # Constants set by BNB and Microboone 
    L = 500. # m
    E_min = 0.1 # GeV # TODO: check with BNB low energy limit
    E_max = 5.0 # GeV
    initial_flux = 10e-4 # nu_mu's/POT/GeV/m^2 # TODO: check with BNB flux
    
    # Generate data points in energy
    E = np.random.normal((E_max+E_min)/2, size=n_points)
    
    # Calculate the probability of oscillation
    P = stheta * np.sin((L/E)*dm)**2
    
    # Calculate the number of muon and electron neutrinos
    N_nue = initial_flux * P
    N_numu = initial_flux * (1-P)
    
    # Create a data frame to hold the data
    data = pd.DataFrame(dict(N_numu = N_numu,
                        N_nue = N_nue,
                        L = L*np.ones_like(N_numu),
                        N_numu_initial = initial_flux*np.ones_like(N_numu), 
                        E = E))
    
    return data

def write_simulated_data(filename, data_dir='data', n_points=100):
    data = generative_model(n_points)
    data.to_csv(data_dir+'/'+file_name, index=False)
    return