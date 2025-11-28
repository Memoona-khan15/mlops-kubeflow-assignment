# src/download_data.py
import pandas as pd
from sklearn.datasets import load_diabetes
import os

# Note: load_boston is deprecated. We'll use diabetes dataset (regression) as substitute.
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['TARGET'] = data.target
os.makedirs('data', exist_ok=True)
df.to_csv('data/raw_data.csv', index=False)
print("Saved data/raw_data.csv")
