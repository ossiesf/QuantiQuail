import unittest
from src.fetcher import Fetcher

class TestFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = Fetcher("SPY")
        self.fetcher.fetch_data()

    def test_fetcher_download(self):
        self.assertIsNotNone(self.fetcher.data, "Data should not be None")

    def test_check_data(self):
        self.fetcher.normalize_data()
        self.fetcher.check_data()
        print("Data checked successfully.")

    def test_normalize_data(self):
        self.fetcher.normalize_data()
        self.fetcher.check_data()
        print("Data normalized successfully.")

if __name__ == "__main__":
    unittest.main()