import yfinance as yf
from sklearn.preprocessing import StandardScaler

class Fetcher:
    def __init__(self, ticker, period="5y"):
        self.ticker = ticker
        self.data = None
        self.period = period

    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.Ticker(self.ticker).history(period=self.period)
        print(f"Data for {self.ticker} fetched successfully.")
        print(f"Data contains {self.data.isnull().sum().sum()} missing values.\n")
        return self.data

    def save_to_csv(self, filename):
        if self.data is not None:
            self.data.to_csv(f'/data/{filename}')
            print(f"Data saved to /data/{filename}")
        else:
            print("No data to save. Please fetch data first.")
            
    def normalize_data(self, columns=None):
        if columns is None:
            columns = ["Open", "High", "Low", "Close", "Volume"]
            
        # Standardize the data with a mean of 0 and standard deviation of 1
        if self.data is not None:
            scaler = StandardScaler()
            self.data[columns] = scaler.fit_transform(self.data[columns])
            self.data[columns] = self.data[columns].round(2)  # Round to 2 decimal places
            print(f"Data for {self.ticker} normalized successfully. Returning cleaned data.")
            return self.clean_data(self.data)
        else:
            print("No data to normalize. Please fetch data first.")
            
    def check_data(self):
        if self.data is not None:
            print("Size of data with missing values: ", self.data.isnull().sum())
            print("Number of duplicates: ", self.data.duplicated().sum())
            
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                desc = self.data[col].describe()
                mean = desc['mean']
                std = desc['std']
                min_val = desc['min']
                max_val = desc['max']
                # Are we within 3 STD from the mean (does the col have an outlier)?
                print(f"Checking {col} for outliers...")
                if min_val < mean - 3 * std or max_val > mean + 3 * std:
                    import matplotlib.pyplot as plt
                    
                    print(f"Possible outlier detected in {col}")
                    print("Data description for outliers: \n", self.data.describe())
                    # Use box plots to summarize distributions and spot outliers
                    # Use line plots (time series) to see how values change over time and spot trends / sudden events
                    self.data.boxplot(column=[col])
                    plt.show()
                    self.data[col].plot(title=f"{self.ticker} {col} over time", figsize=(12, 6))
                    plt.xlabel("Date")
                    plt.ylabel(col)
                    plt.show()
                else: 
                    print("Data checked successfully.")
        else:
            print("No data available. Please fetch data first.")
            
    def clean_data(self, data):
        if data is not None:
            # Clean the data
            import numpy as np
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.dropna().reset_index(drop=True)
            return data