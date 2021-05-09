import pandas as pd
import random
import csv
import numpy as np
import os
import scipy.stats as stats

class Data:
    """
    This class can simulate or load data for a neutrino disappearnce
    short beam line experiment. 
    """
    
    def __init__(self, N_numu, N_nue, E): 
        """
        Inputs
            N_numu: int or numpy array of int
                Number of muon neutrinos shot at the detector per energy bin.
            N_nue: int or numpy array of int int
                Number of electron neutrinos seen at the detector per energy bin.
            E: float or numpy array of floats
                Center of energy bins of the incoming muon neutrinos in GeV
                
        Returns
            None
        """
        self.N_numu = N_numu
        self.N_nue = N_nue
        self.E = E
        
    def __repr__(self):
        """
        Formats the data object for print statements
        """
        print_str = 'E: '+str(self.E)+'\n'
        print_str += 'N_numu: '+str(self.N_numu)+'\n'
        print_str += 'N_nue: '+str(self.N_nue)
        return print_str
        
    @classmethod
    def load(self, filename, data_dir='data'):
        """
        Creates a Data object based on input from a csv file.
        
        Inputs
            filename: string
                The full path and name of the file to be loaded. Must
                be in the format of pandas data frame with three
                columns named N_numu, N_nue, and E
                
        Returns
            A Data object
        """
        filepath = get_data_file_path(filename)
        data = pd.read_csv(filepath)
        
        return Data(data.N_numu, data.N_nue, data.E)
        
    @classmethod   
    def simulate_detector(self, ss2t, dms, 
                          N_numu = np.array([600, 6000, 60000, 600000]), 
                          E_bin_edges = np.array([0.01, 0.05, 1.5, 2.1, 3.0]), 
                          mu_L=0.5, sigma_L=.025):
        """
        Creates a Data object with simulated data based on parameters
        given for some detector. Defaults are set to match the
        Microboone detector.
        
        Inputs 
            ss2t: float between 0 and 1
                The oscillation paramter sin^2(2*theta)
            dms: float >= 0
                The oscillation parameter delta m^2 (squared mass difference)
            N_numu: int or numpy array of int
                Number of muon neutrinos shot at the detector per energy bin.
            E_bin_edges: float or numpy array of floats
                Edges of energy bins in GeV. Must be len(N_numu)+1 in size
            mu_L: float >= 0 in meters
                The detector baseline (distance from neutrino beam). For now,
                I am considering the basline as the average distance traveled
                by the muon neutrinos. 
            sigma_L: float
                Standard deviation of beamline, L
            
        Returns
            A Data object
        """
        N_nue = None
        if isinstance(E_bin_edges, np.ndarray) and isinstance(N_numu, np.ndarray):
            if len(E_bin_edges) == len(N_numu)+1:
                mu_E, sigma_E = GetEnergies(E_bin_edges)
                N_nue = []
                for i in range(len(N_numu)):
                    N_nue.append(self.simulate_data(self, N_numu[i], ss2t, dms, 
                                                    mu_L, mu_E[i], sigma_L, sigma_E[i]))
            else:
                raise Exception('Size of input arrays to not match. E_bin_edges must have len(N_numu)+1')
        elif isinstance(E_bin_edges, float) and isinstance(N_numu, int):
            mu_E = E_bin_edges
            sigma_E = 0.01
            N_nue = self.simulate_data(self, N_numu, ss2t, dms, 
                                mu_L, mu_E, sigma_L, sigma_E)
        else:
            raise Exception('Invalid input') 

        return Data(N_numu, N_nue, mu_E)

    def simulate_data(self, N_numu, ss2t, dms, mu_L=0.5, mu_E=1.0, 
                      sigma_L=.025, sigma_E=0.25):
        """
        Simulates data of how many muon neutrinos oscillate to electron 
        neutrinos based on given parameters for the experiment detector 
        and beamline. 
        
        Inputs
            N_numu: int
                Number of muon neutrinos shot at the detector. 
            ss2t: float between 0 and 1
                The oscillation paramter sin^2(2*theta)
            dms: float >= 0
                The oscillation parameter delta m^2 (squared mass difference)
            mu_L: float >= 0 in km
                The detector baseline (distance from neutrino beam). For now,
                I am considering the basline as the average distance traveled
                by the muon neutrinos. 
            mu_E: float >= 0 in GeV
                The average energy of incoming muon neutrinos.
            sigma_L: float
                Standard deviation of beamline, L
            sigma_E: float
                Standard deviation of muon neutrino energy, E
            
        Returns
            N_nue: integer
                The number of electron neutrinos
        """
        L = stats.norm.rvs(mu_L, sigma_L, N_numu)
        E = stats.norm.rvs(mu_E, sigma_E, N_numu)
        P = OscProbability(ss2t, dms, L, E)
        r = np.random.random(N_numu)
        
        N_nue = sum(list(map(Oscillate, r, P)))
                
        return N_nue
    
    def write_data(self, filename, data_dir='data'):
        """
        Writes data to a csv file as one line with two 
        numbers (N_numu, N_nue)
        
        Inputs
            filename: string
                Name of file to be written.
            data_dir: string
                Name of directory to output the file.
            
        Returns
            None
        """
        data_dir = os.path.join(get_data_dir(data_dir))
        file_path = os.path.join(data_dir, filename)
        
        if isinstance(self.N_numu, int):
            df = pd.DataFrame({'E': self.E, 
                            'N_numu':self.N_numu, 
                            'N_nue':self.N_nue}, index=[0])
        else:
            df = pd.DataFrame({'E': self.E, 
                                'N_numu':self.N_numu, 
                                'N_nue':self.N_nue})
        df.to_csv(file_path, index=False)
        return
    
def OscProbability(ss2t, dms, L, E):
    """
    Returns the oscillation probability of a muon neutrino to an 
    electron neutrino for given experiment and oscillation parameters.

    Inputs 
        ss2t: float or numpy array between 0 and 1
            The oscillation paramter sin^2(2*theta)
        dms: float or numpy array >= 0
            The oscillation parameter delta m^2 (squared mass difference)
        L: float or numpy array >= 0 in km
            Distance the muon neutrino traveled.
        E: float or numpy array >= 0 in GeV
            Energy of incoming muon neutrino.

    Returns
        Oscillation probability (float between 0 and 1)

    """
    return ss2t * np.sin((1.27*L/E)*dms)**2

def get_data_file_path(filename, data_dir='data'):
    """
    Returns file path given a data directory.
    
    Inputs
        filename (str): name of the file
        data_dir (str): name of data directory
    
    Returns
        The absolute file path: str
    """
    # __file__ is the location of the source file currently in use
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    data_dir = os.path.join(start_dir, data_dir)
    return os.path.join(start_dir, data_dir, filename)

def get_data_dir(data_dir='data'):
    """
    Returns the data directory.
    
    Inputs
        data_dir: str
            Name of data directory
    
    Returns
        The data directory path: str
    """
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    return os.path.join(start_dir, data_dir)

def GetEnergies(E_bin_edges):
    """
    Given energy bin edges, finds the mean and width of
    the energy bins.
    
    Inputs
        E_bin_edges: numpy array of floats
            The energy bin edges in GeV
    
    Returns
        mu_E: numpy array of floats
            The mean energy of each bin
        sigma_E: numpy array of floats
            The width of each bin
    """
    
    E_bin_edges = np.array(E_bin_edges)
    mu_E = np.array([0.5*(E_bin_edges[i]+E_bin_edges[i+1]) for i in range(len(E_bin_edges)-1)])
    sigma_E = np.diff(E_bin_edges)
    
    return mu_E, sigma_E

def Oscillate(r, P):
    if r<P:
        return 1
    else:
        return 0
