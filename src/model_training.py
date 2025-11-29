import mlflow
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("artifacts", exist_ok=True)

    model_path = "artifacts/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    metrics_path = "artifacts/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc}, f)
    mlflow.log_artifact(metrics_path)

    return acc
