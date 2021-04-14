from unittest import TestCase
from bayboone.data_io import generative_model
import pandas as  pd

class TestIo(TestCase):
    def test_data_io(self):
        data = generative_model(10) 
        assert data.L[0] == 500
        
        # TODO: add more tests
        
