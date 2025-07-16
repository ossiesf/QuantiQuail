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
        
    def test_relative_strength_index(self):
        self.data = self.features.relative_strength_index(self.data)
        print("Sample of data with RSI feature:\n", self.data[:15], "\n")
        
        self.assertIn('RSI', self.data.columns, "RSI column should exist")
        self.assertEqual(self.data['RSI'].isnull().sum(), 0, "RSI should not have NaN values")