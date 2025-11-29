import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(X_train, y_train, X_test, y_test):
    mlflow.set_experiment("MLflow-Pipeline")

    with mlflow.start_run(run_name="train_model"):

        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metric
        mlflow.log_metric("accuracy", acc)

        # Save model locally
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        # Log artifact
        mlflow.log_artifact(model_path)

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="BreastCancerModel")

        print(f"Model trained. Accuracy: {acc}")
        return acc
