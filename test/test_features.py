from src.features import Features
from src.fetcher import Fetcher
import unittest

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        self.fetcher = Fetcher('SPY', "1y")
        self.data = self.fetcher.fetch_data()
        self.features = Features()

    def test_daily_returns(self):
        self.data = self.features.daily_returns(self.data)
        print("Sample of data with feature and label:\n", self.data.head(), "\n")
        print("Sample of data that does not meet the threshold:\n", self.data[self.data['Label'] == -1].head(), "\n")
        
        self.assertIn('Daily Returns', self.data.columns, "Daily Returns column should exist")
        self.assertIn('Label', self.data.columns, "Label column should exist")

        self.assertEqual(self.data['Daily Returns'].isnull().sum(), 0,
                         "Daily returns should not NaN values")
        self.assertEqual(self.data['Label'].isnull().sum(), 0, 
                         "Labels should not NaN values")