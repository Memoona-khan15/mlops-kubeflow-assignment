import pandas as pd
import mlflow
from src.preprocess_data import preprocess_data
from src.train_model import train_model
from src.download_data import create_dataset

def run_pipeline():
    print("STEP 1: Extracting data...")
    create_dataset()

    df = pd.read_csv("data/raw_data.csv")

    print("STEP 2: Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("STEP 3: Training...")
    accuracy = train_model(X_train, y_train, X_test, y_test)

    print("STEP 4: Completed. Accuracy:", accuracy)


if __name__ == "__main__":
    run_pipeline()
