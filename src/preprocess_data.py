import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def preprocess_data(df, target="label"):
    mlflow.set_experiment("MLflow-Pipeline")
    with mlflow.start_run(run_name="preprocess"):

        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found. Columns = {list(df.columns)}")

        X = df.drop(columns=[target])
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(scaler, "artifacts/scaler.pkl")
        mlflow.log_artifact("artifacts/scaler.pkl")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
