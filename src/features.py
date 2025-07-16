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
        data = data[data['Label'] != -1].reset_index(drop=True)
        print(f"Daily returns feature added with {data['Label'].value_counts().to_dict()}")
        return data
        
    # Moving averages TODO

    # RSI TODO