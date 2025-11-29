from src.extract_data import extract_data
from src.preprocess_data import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model

def run_pipeline():
    print("STEP 1: Extracting data...")
    df = extract_data()

    print("STEP 2: Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("STEP 3: Training...")
    model = train_model(X_train, y_train)

    print("STEP 4: Evaluating...")
    mse, r2 = evaluate_model(model, X_test, y_test)

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
    print("💡 MSE:", mse)
    print("💡 R2 Score:", r2)

if __name__ == "__main__":
    run_pipeline()
