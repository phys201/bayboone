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

def generative_model(n_points=100):
    # TODO: add doc strings, add comments, edit variable names to not be so ghastly 
    
    L = 500. # m
    E_min = 0.1 # GeV # TODO: check with BNB low energy limit
    E_max = 5.0 # GeV
    initial_flux = 10e-4 # nu_mu's/POT/GeV/m^2 # TODO: check with BNB flux
    
    sin2_theta12 = 0.5 # TODO: think of a more reasonable guess (starting big so its easier to see)
    dm2_14 = 10e-5 # GeV^2
    
    E = np.random.normal((E_max+E_min)/2, size=n_points)
    
    P = sin2_theta12 * np.sin((L/E)*dm2_14)**2
    
    N_nue = initial_flux * P
    N_numu = initial_flux * (1-P)
    
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