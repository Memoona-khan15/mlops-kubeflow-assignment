import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import joblib
import os

def preprocess_data(df):
    target = "label"

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/scaler.pkl"

    joblib.dump(scaler, path)
    mlflow.log_artifact(path)

    return X_train, X_test, y_train, y_test
