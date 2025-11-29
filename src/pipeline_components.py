from kfp import dsl
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1. DATA EXTRACTION
# -------------------------
@dsl.component(
    base_image="python:3.11",
    output_component_file="components/extract_data.yaml"
)
def extract_data(data_path: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_data_path = os.path.join(project_root, data_path)
    return raw_data_path

# -------------------------
# 2. DATA PREPROCESSING
# -------------------------
from typing import NamedTuple

@dsl.component(
    base_image="python:3.11",
    output_component_file="components/preprocess_data.yaml"
)
def preprocess_data(data_path: str) -> NamedTuple('Outputs', [
    ('X_train', str),
    ('X_test', str),
    ('y_train', str),
    ('y_test', str),
    ('scaler', str)
]):
    import pandas as pd
    import os
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    target_column = "MEDV"
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    pd.DataFrame(X_train).to_csv("artifacts/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("artifacts/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("artifacts/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("artifacts/y_test.csv", index=False)

    from collections import namedtuple
    outputs = namedtuple('Outputs', ['X_train', 'X_test', 'y_train', 'y_test', 'scaler'])
    return outputs(
        X_train="artifacts/X_train.csv",
        X_test="artifacts/X_test.csv",
        y_train="artifacts/y_train.csv",
        y_test="artifacts/y_test.csv",
        scaler="artifacts/scaler.pkl"
    )

# -------------------------
# 3. MODEL TRAINING
# -------------------------
@dsl.component(
    base_image="python:3.11",
    output_component_file="components/train_model.yaml"
)
def train_model(X_train_path: str, y_train_path: str) -> str:
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/random_forest_model.pkl"
    joblib.dump(model, model_path)
    return model_path

# -------------------------
# 4. MODEL EVALUATION
# -------------------------
from typing import NamedTuple

@dsl.component(
    base_image="python:3.11",
    output_component_file="components/evaluate_model.yaml"
)
def evaluate_model(model_path: str, X_test_path: str, y_test_path: str) -> NamedTuple('Metrics', [
    ('mse', float),
    ('r2', float)
]):
    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    model = joblib.load(model_path)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    from collections import namedtuple
    Metrics = namedtuple('Metrics', ['mse', 'r2'])
    return Metrics(mse, r2)


