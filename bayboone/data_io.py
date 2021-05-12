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
        """
        # Check inputs
        if isinstance(N_numu, int) and isinstance(N_nue, int) and isinstance(E, float):
            if N_numu<=0 or N_nue<0 or N_numu<=N_nue: 
                raise ValueError('N_numu must be less than N_nue, N_numu must be greater than zero, N_nue must be equal or greater than zero.')
            if E <= 0.0:
                raise ValueError('Energy must be greater than 0')
        elif isinstance(N_numu, np.ndarray) and isinstance(N_nue, np.ndarray) and isinstance(E, np.ndarray):
            if len(N_numu) != len(N_nue) or len(N_nue) != len(E):
                raise ValueError('N_numu, N_nue, and E must have same length')
                
            elif (N_numu <= 0).any() or (N_nue < 0).any() or (E <= 0.0).any():
                raise ValueError('Values must be greater or equal to zero. N_numu cannot be zero')
                
            elif N_numu.dtype != np.dtype('int64'):
                raise ValueError('N_numu must contain all integers')
        else:
            raise ValueError('N_numu, N_nue, and E must be single int, int, float or numpy arrays')
            
        
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
        
        col_names = np.array(['E', 'N_numu', 'N_nue'])
        if not np.array_equal(np.array(data.columns), col_names):
            raise ValueError('Data file not in correct format')
            
        if len(data.N_numu) > 1:
            N_numu = np.array(data.N_numu)
            N_nue = np.array(data.N_nue)
            E = np.array(data.E)
        else:
            N_numu = int(data.N_numu)
            N_nue = int(data.N_nue)
            E = float(data.E)

        return Data(N_numu, N_nue, E)
        
    @classmethod   
    def simulate_detector(self, ss2t, dms, 
                          N_numu = np.array([60, 600, 6000, 60000, 600000]), 
                          E_bin_edges = np.array([0.01, 0.4, 0.6, 1.0, 1.5, 2]), 
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

        # Checking correct type of input
        is_array = False
        if isinstance(E_bin_edges, np.ndarray) and isinstance(N_numu, np.ndarray):
            is_array = True
            if len(E_bin_edges) != len(N_numu)+1:
                raise ValueError('E_bin_edges must be len(N_numu)+1 in length')
            elif N_numu.dtype != np.dtype('int64'):
                raise ValueError('N_numu must contain all integers')
        elif not isinstance(N_numu, int):
            raise ValueError('N_numu must be integer')
        
        N_nue = None
        if is_array:
            mu_E, sigma_E = GetEnergies(E_bin_edges)
            N_nue = []
            for i in range(len(N_numu)):
                N_nue.append(self.simulate_data(self, ss2t, dms, int(N_numu[i]), 
                                                mu_L, mu_E[i], sigma_L, sigma_E[i]))
            N_nue = np.array(N_nue)

        else:
            mu_E = E_bin_edges
            sigma_E = .1
            N_nue = self.simulate_data(self, ss2t, dms, N_numu,
                                mu_L, mu_E, sigma_L, sigma_E) 

        return Data(N_numu, N_nue, mu_E)

    def simulate_data(self, ss2t, dms, N_numu, mu_L=0.5, mu_E=1.0, sigma_L=.025, sigma_E=0.25):
        """
        Simulates data of how many muon neutrinos oscillate to electron 
        neutrinos based on given parameters for the experiment detector 
        and beamline. 
        
        Inputs
            ss2t: float between 0 and 1
                The oscillation paramter sin^2(2*theta)
            dms: float >= 0
                The oscillation parameter delta m^2 (squared mass difference)
            N_numu: int 
                Number of muon neutrinos shot at the detector. 
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

        # Check values that are not checked elsewhere
        if N_numu <= 0 or not isinstance(N_numu, int):
            raise ValueError('N_numu must be an integer greater than 0')
        
        if mu_L < 0.0:
            raise ValueError('mu_L must be equal or greater than 0')
        
        if mu_E <=0.0:
            raise ValueError('mu_E must be greater than 0')
  
        if sigma_L<=0.0:
            raise ValueError('sigma_L must be float equal or greater than0.0')
            
        if sigma_E<=0.0:
            raise ValueError('sigma_E must be float equal or greater than 0.0')
            
        L = stats.truncnorm.rvs(a=(-mu_L)/sigma_L, b=np.inf, loc=mu_L, scale=sigma_L, size=N_numu) 
        E = stats.truncnorm.rvs(a=(-mu_E)/sigma_E, b=np.inf, loc=mu_E, scale=sigma_E, size=N_numu) 
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
        L: float >= 0 in km
            Distance the muon neutrino traveled.
        E: float or numpy array >= 0 in GeV
            Energy of incoming muon neutrino.

    Returns
        Oscillation probability (float between 0 and 1)

    """
    if ss2t<0.0 or ss2t>1.0:
            raise ValueError('ss2t must be float between 0.0 and 1.0')

    if dms<0.0:
        raise ValueError('dms must be float equal or greater than 0.0')
        
    if not isinstance(L, np.ndarray):
        if L<0.0:
            raise ValueError('L must be float equal or greater tna 0.0')
    elif (L<=0.0).any():
        raise ValueError('All values of L must be greater than or equal to 0.0')
        
    if not isinstance(E, np.ndarray):
        if E <= 0.0:
            raise ValueError('E must be greater than 0.0')
    elif (E<=0.0).any():
        raise ValueError('All values of E must be greater than 0.0')
    
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
    if not isinstance(E_bin_edges, np.ndarray):
        raise ValueError('E_bin_edges must be numpy array')
    if (E_bin_edges <= 0.0).any():
        raise ValueError('No energy bin edges can be less than 0.0')
    if (np.diff(E_bin_edges) <= 0.0).all():
        raise ValueError('Energy bin edges must be in increasing order')
    
    E_bin_edges = np.array(E_bin_edges)
    mu_E = np.array([0.5*(E_bin_edges[i]+E_bin_edges[i+1]) for i in range(len(E_bin_edges)-1)])
    sigma_E = np.diff(E_bin_edges)
    
    return mu_E, sigma_E

def Oscillate(r, P):
    """
    Oscillated based on random number r and 
    probability P
    
    Inputs
        r: float between 0 and 1
            random number
        P: float between 0 and 1
            Probability of oscillation
    Returns
        0 if no oscillation, 1 if oscillated
    """
    if r<P:
        return 1
    else:
        return 0

