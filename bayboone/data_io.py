import pandas as pd
import random
import csv
import numpy as np
import os

class Data:
    """
    This class can simulate or load data for a neutrino disappearnce
    short beam line experiment. 
    """
    
    def __init__(self, N_numu, N_nue, E): 
        """
        Inputs
            N_numu: array of int
                Number of muon neutrinos shot at the detector per energy bin.
            N_nue: array of int int
                Number of electron neutrinos seen at the detector per energy bin.
            E: array of floats
                Center of energy bins of the incoming muon neutrinos in GeV
                
        Returns
            None
        """
        self.N_numu = N_numu
        self.N_nue = N_nue
        self.E = E
        
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
                          N_numu = [600, 6000, 60000, 600000], 
                          E_bin_edges = [0.01, 0.05, 1.5, 2.1, 3.0], 
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
            N_numu: array of int
                Number of muon neutrinos shot at the detector per energy bin.
            E_bin_edges: array of floats
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
        mu_E, sigma_E = GetEnergies(E_bin_edges)
        
        N_nue = []
        for i in range(len(N_numu)):
            N_nue.append(self.simulate_data(self, N_numu[i], ss2t, dms, 
                                            mu_L, mu_E[i], sigma_L, sigma_E[i]))
        
        return Data(N_numu, N_nue, mu_E)

    def simulate_data(self, N_numu, ss2t, dms, mu_L=0.5, mu_E=1.0, 
                      sigma_L=.025, sigma_E=0.25, random_seed=True):
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
        if not random_seed:
            np.random.seed(random_seed)
        
        N_nue = 0
        for _ in range(N_numu):
            L = random.gauss(mu_L, sigma_L)
            E = random.gauss(mu_E, sigma_E)
            P = OscProbability(ss2t, dms, L, E)
            r = np.random.random()

            if r < P:
                N_nue += 1
                
        print('Data: ',N_numu, N_nue)
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
        ss2t: float between 0 and 1
            The oscillation paramter sin^2(2*theta)
        dms: float >= 0
            The oscillation parameter delta m^2 (squared mass difference)
        L: float >= 0 in km
            Distance the muon neutrino traveled.
        E: float >= 0 in GeV
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
        E_bin_edges: array of floats
            The energy bin edges in GeV
    
    Returns
        mu_E: array of floats
            The mean energy of each bin
        sigma_E: array of floats
            The width of each bin
    """
    
    E_bin_edges = np.array(E_bin_edges)
    mu_E = [0.5*(E_bin_edges[i]+E_bin_edges[i+1]) for i in range(len(E_bin_edges)-1)]
    sigma_E = np.diff(E_bin_edges)
    
    return mu_E, sigma_E
