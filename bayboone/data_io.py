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
    
    def __init__(self, N_numu, N_nue): 
        """
        Inputs
        ------
            N_numu: int
                Number of muon neutrinos shot at the detector.
            N_nue: int
                Number of electron neutrinos seen at the detector. 
        """
        self.N_numu = N_numu
        self.N_nue = N_nue
        
    @classmethod
    def load(self, filename, data_dir='data'):
        """
        Creates a Data object based on input from a csv file.
        Assumes format of file is one line with two numbers (N_numu, N_nue)
        
        Inputs
        ------
            filename: string
                The full path and name of the file to be loaded. 
        """
        filepath = get_data_file_path(filename)
        with open(filepath, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                N_numu = int(row[0])
                N_nue = int(row[1])
        
        return Data(N_numu, N_nue)
        
    @classmethod
    def simulate_microboone(self, N_numu, ss2t, dms):
        """
        Creates a Data object with simulated data based on parameters
        from the microboone detector.
        
        Inputs
        ------
        N_numu: int
            Number of muon neutrinos shot at the detector. 
        ss2t: float between 0 and 1
            The oscillation paramter sin^2(2*theta)
        dms: float >= 0
            The oscillation parameter delta m^2 (squared mass difference)
        """
        N_numu, N_nue = self.simulate_data(self, N_numu, ss2t, dms)
        return Data(N_numu, N_nue)
    
    @classmethod   
    def simulate_detector(self, N_numu, ss2t, dms, mu_L, mu_E, sigma_L, sigma_E):
        """
        Creates a Data object with simulated data based on parameters
        given for some detector.
        
        Inputs
        ------
        N_numu: int
            Number of muon neutrinos shot at the detector. 
        ss2t: float between 0 and 1
            The oscillation paramter sin^2(2*theta)
        dms: float >= 0
            The oscillation parameter delta m^2 (squared mass difference)
        mu_L: float >= 0 in meters
            The detector baseline (distance from neutrino beam). For now,
            I am considering the basline as the average distance traveled
            by the muon neutrinos. 
        mu_E: float >= 0 in GeV
            The average energy of incoming muon neutrinos. 
        """
        N_numu, N_nue = self.simulate_data(self, N_numu, ss2t, dms, mu_L, mu_E, sigma_L, sigma_E)
        return Data(N_numu, N_nue)
    
    def write_data(self, filename, data_dir='data'):
        """
        Writes data to a csv file as one line with two 
        numbers (N_numu, N_nue)
        
        Inputs
        ------
        filename: string
            Name of file to be written.
        data_dir: string
            Name of directory to output the file.
        """
        data_dir = os.path.join(get_data_dir(data_dir))
        file_path = os.path.join(data_dir, filename)
        with open(file_path, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.N_numu,self.N_nue])
        return


    def simulate_data(self, N_numu, ss2t, dms, mu_L=0.5, mu_E=1.0, sigma_L=.025, sigma_E=0.25, random_seed=True):
        """
        Simulates data of how many muon neutrinos oscillate to electron neutrinos based
        on given parameters for the experiment detector and beamline. 
        
        Inputs
        ------
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
        return N_numu, N_nue
    
def OscProbability(ss2t, dms, L, E):
    """
    Returns the oscillation probability of a muon neutrino to an 
    electron neutrino for given experiment and oscillation parameters.

    Inputs 
    ------
    ss2t: float between 0 and 1
            The oscillation paramter sin^2(2*theta)
        dms: float >= 0
            The oscillation parameter delta m^2 (squared mass difference)
        L: float >= 0 in km
            Distance the muon neutrino traveled.
        E: float >= 0 in GeV
            Energy of incoming muon neutrino.

    Returns
    -------
        Oscillation probability (float between 0 and 1)

    """
    return ss2t * np.sin((1.27*L/E)*dms)**2

def get_data_file_path(filename, data_dir='data'):
    """
    Returns file path given a data directory.
    
    Inputs
    ------
    filename (str): name of the file
    data_dir (str): name of data directory
    
    Returns
    -------
        The absolute file path
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
    -------
    data_dir (str): name of data directory
    
    Returns
    -------
        The data directory path
    """
    start = os.path.abspath(__file__)
    start_dir = os.path.dirname(start)
    return os.path.join(start_dir, data_dir)
