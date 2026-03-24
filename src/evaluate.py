"""
Final evaluation against the held-out test set.

Run only after training is complete:
    python src/evaluate.py

Loads the saved model and feature column list, then scores against data/test.csv.
This simulates how the model would perform on unseen production data.
"""

import json
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_data, select_features

TEST_DATA_PATH = "data/test.csv"
MODEL_PATH = "models/lgbm_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.json"


def load_test_data(feature_cols: list, path: str = TEST_DATA_PATH):
    df = load_data(path)
    X_test, y_test, _ = select_features(df, feature_cols=feature_cols)
    return X_test, y_test


def load_artifacts(model_path: str = MODEL_PATH, cols_path: str = FEATURE_COLS_PATH):
    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_cols = json.load(f)
    print(f"Loaded model from {model_path}")
    print(f"Loaded {len(feature_cols)} feature columns from {cols_path}")
    return model, feature_cols


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n--- Final Test Set Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


if __name__ == "__main__":
    model, feature_cols = load_artifacts()
    X_test, y_test = load_test_data(feature_cols)
    evaluate(model, X_test, y_test)
