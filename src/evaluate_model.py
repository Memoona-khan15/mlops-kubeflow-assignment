import mlflow
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    mlflow.set_experiment("MLflow-Pipeline")
    with mlflow.start_run(run_name="evaluate_model"):
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        return mse, r2
