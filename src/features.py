class Features:
    # Daily returns
    # We need a label to target based off of this feature for regression:
    # 1 for up prediction, 0 for down
    def daily_returns(self, data):
        data = data.copy()  # Avoid modifying the original DataFrame
        data['Daily Returns'] = data['Close'].pct_change().round(4)
        next_returns = data['Daily Returns'].shift(-1)
        
        # Some returns are so close to zero that they are not significant,
        # so we will use a threshold to avoid noise
        threshold = 0.001
        data['Label'] = -1 # Default label for insignificant returns
        data.loc[next_returns > threshold, 'Label'] = 1
        data.loc[next_returns < -threshold, 'Label'] = 0
        data = data[data['Label'] != -1].dropna().reset_index(drop=True)
        print(f"Daily returns feature added with {data['Label'].value_counts().to_dict()}")
        return data
        
    # Relative Strength Index TODO
    def relative_strength_index(self, data, period=14):
        data = data.copy()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi.round(2)
        data = data.iloc[period:].reset_index(drop=True)  # Drop the first 'period' NaN values
        return data
    
    def find_rsi_period(self, data, max_period=30):
        # Find the best RSI period based on correlation with daily returns
        
        # Requires we have the 'Daily Returns' column already calculated
        if 'Daily Returns' not in data.columns:
            raise ValueError("Daily Returns must be calculated before finding RSI period.")
        
        correlations = {}
        for period in range(2, max_period + 1):
            rsi = self.relative_strength_index(data.copy(), period)['RSI']
            correlation = data['Daily Returns'].corr(rsi)
            correlations[period] = correlation
            
        best_period = max(correlations, key=correlations.get)
        print(f"Best RSI period found: {best_period} with correlation {correlations[best_period]:.4f}")
        return best_period
    
    # Moving averages TODO

    # Bollinger Bands TODO