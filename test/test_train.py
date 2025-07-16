import unittest
from src.fetcher import Fetcher
from src.features import Features
from src.train import Train
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
class TestTrain(unittest.TestCase):
    
    def setUp(self):
        self.fetcher = Fetcher("SPY")
        self.features = Features()
        self.fetcher.fetch_data()
        self.data = self.fetcher.normalize_data()
        self.data = self.features.daily_returns(self.data)
        self.data = self.data.dropna()  # Ensure no NaNs remain after feature engineering
        self.train = Train('Label', self.data)

    def test_data_split(self):
        X_train, X_test, y_train, y_test = self.train.train_test_split()
        print("Train-test split completed. Running tests...")
        n = self.data.shape[0]
        expected_train = int(0.8 * n)
        expected_test = n - expected_train

        self.assertAlmostEqual(X_train.shape[0], expected_train, delta=10,
                               msg="Training set size should be approximately 80% of the total data")
        self.assertAlmostEqual(X_test.shape[0], expected_test, delta=10,
                               msg="Test set size should be approximately 20% of the total data")
        self.assertAlmostEqual(y_train.shape[0], expected_train, delta=10,
                               msg="y_train should have the same number of samples as X_train")
        self.assertAlmostEqual(y_test.shape[0], expected_test, delta=10,
                               msg="y_test should have the same number of samples as X_test")
        
        self.assertTrue(X_train.isna().sum().sum() == 0,
                        "X_train should not contain NaN values")
        self.assertTrue(X_test.isna().sum().sum() == 0,
                        "X_test should not contain NaN values")
        self.assertTrue(y_train.isna().sum() == 0,
                        "y_train should not contain NaN values")
        self.assertTrue(y_test.isna().sum() == 0,
                        "y_test should not contain NaN values")
        
        print("All tests passed successfully, test and train sets created.")
        
    def test_find_best_params(self):
        X_train, _, y_train, _ = self.train.train_test_split()
        print("Shape of data for training:", X_train.shape, y_train.shape)
        self.assertEqual(X_train.shape[0], y_train.shape[0],
                         "X_train and y_train should have the same number of samples")
        
        best_model = self.train.find_best_params(X_train, y_train)
        self.assertIsNotNone(best_model, "Best model should not be None")
        
    def test_predict(self):
        X_train, X_test, y_train, y_test = self.train.train_test_split()
        best_model = self.train.find_best_params(X_train, y_train)
        
        predictions = self.train.predict(best_model, X_test)
        print("Predictions looked like this:\n", predictions[:5])
        self.assertEqual(len(predictions), len(X_test),
                         "Predictions should have the same number of samples as X_test")
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", class_report)
        print("Predictions made and evaluated successfully.")