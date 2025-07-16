from src.features import Features
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Train:
    def __init__(self, label, data):
        self.features = Features()
        self.label = label
        self.data = self.features.daily_returns(data) if data is not None else None
        self.data = self.features.relative_strength_index(self.data) if self.data is not None else None

    def train_test_split(self):
        if self.data is not None:
            X = self.data.drop(self.label, axis=1).replace([np.inf, -np.inf], np.nan)
            y = self.data[self.label].replace([np.inf, -np.inf], np.nan)

            # Drop rows with any NaN in X or y
            valid_idx = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
            
            from sklearn.model_selection import train_test_split
            train_test_split(X, y, test_size=0.2, shuffle=False)
            return train_test_split(X, y, test_size=0.2, shuffle=False)
        else:
            print("No data available for training. Please fetch data first.")
            
    def find_best_params(self, X_train, y_train):
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [10, 20, 50, 100],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Remove inf/-inf and values outside a reasonable range
        mask = np.isfinite(y_train) & (np.abs(y_train) < 1e6)
        X_train_clean = X_train[mask]
        y_train_clean = y_train[mask]
        
        print(y_train_clean.head())
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
        grid_search.fit(X_train_clean, y_train_clean)

        print("Best parameters found: ", grid_search.best_params_)
        return grid_search.best_estimator_
    
    def train_model(self, X_train, y_train):
        model = RandomForestClassifier(random_state=42)
        params = self.find_best_params(X_train, y_train)
        model.set_params(**params.get_params())
        model.fit(X_train, y_train)
        print("Model trained successfully with parameters: ", model.get_params())
        return model
    
    def predict(self, model, X_test):
        if model is not None and X_test is not None:
            predictions = model.predict(X_test)
            print("Predictions made successfully.")
            return predictions
        else:
            print("Model or test data is not available for prediction.")
            return None