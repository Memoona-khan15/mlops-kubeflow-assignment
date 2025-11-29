import mlflow
import pandas as pd
import os

def extract_data():
    # Create data folder
    os.makedirs("data", exist_ok=True)

    # Load dataset
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.rename(columns={"target": "label"}, inplace=True)

    df.to_csv("data/raw_data.csv", index=False)

    # Log dataset file (safe)
    mlflow.log_artifact("data/raw_data.csv")

    print("Saved data/raw_data.csv with 'label' column.")
    return df
