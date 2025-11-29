import mlflow
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model inside artifacts folder
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.pkl"
    joblib.dump(model, model_path)

    # Log model
    mlflow.log_artifact(model_path)

    # Save accuracy
    metrics_path = "artifacts/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)

    # Log metrics file
    mlflow.log_artifact(metrics_path)

    return accuracy
