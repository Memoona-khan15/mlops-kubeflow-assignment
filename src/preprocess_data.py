import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import os
import json

def preprocess_data(df):
    target = "label"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. CSV columns = {list(df.columns)}")

    # Split features/target
    X = df.drop(columns=[target])
    y = df[target]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create safe artifact directory
    os.makedirs("artifacts", exist_ok=True)

    # Save scaler
    scaler_path = "artifacts/scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Log safely
    mlflow.log_artifact(scaler_path)

    return X_train, X_test, y_train, y_test
