import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_model(X_train, y_train):
    mlflow.set_experiment("MLflow-Pipeline")
    with mlflow.start_run(run_name="train_model"):
        mlflow.sklearn.autolog()

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        mlflow.log_artifact("artifacts/model.pkl")

        return model
