import mlflow
from src.extract_data import extract_data
from src.preprocess_data import preprocess_data
from src.model_training import train_model

def run_pipeline():
    print("STEP 1: Extracting data...")
    df = extract_data()

    print("STEP 2: Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("STEP 3: Training model...")
    acc = train_model(X_train, y_train, X_test, y_test)

    print("Pipeline finished. Accuracy:", acc)

if __name__ == "__main__":
    mlflow.set_experiment("MLflow-Pipeline")
    with mlflow.start_run(run_name="full_pipeline"):

        run_pipeline()

        # End safe
        mlflow.end_run()
