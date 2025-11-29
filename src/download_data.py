import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def create_dataset():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["label"] = data.target

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_data.csv", index=False)
    print("Saved data/raw_data.csv with 'label' column.")

if __name__ == "__main__":
    create_dataset()
