import mlflow
import pandas as pd

def extract_data(path="data/raw_data.csv"):
    mlflow.set_experiment("MLflow-Pipeline")
    with mlflow.start_run(run_name="extract_data"):
        df = pd.read_csv(path)
        mlflow.log_param("rows", df.shape[0])
        mlflow.log_param("columns", list(df.columns))
        mlflow.log_artifact(path)
        return df
