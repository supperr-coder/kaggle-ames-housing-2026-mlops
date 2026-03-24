"""
Train a LightGBM regressor and evaluate on the validation set.
Each run is tracked as an MLflow experiment.

Usage:
    python src/train.py

View results:
    mlflow ui        (then open http://127.0.0.1:5000)
"""

import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_squared_error, r2_score

from preprocess import load_data, select_features, split_data

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm_model.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_cols.json")

EXPERIMENT_NAME = "ames-housing"

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50
LOG_PERIOD = 100


def train(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=LOG_PERIOD),
    ]

    model = lgb.train(
        PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    print(f"Best iteration: {model.best_iteration}")
    return model


def evaluate(model, X, y, split_name: str = "val"):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"\n{split_name} RMSE: {rmse:.4f}")
    print(f"{split_name} R²:   {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


def feature_importance(model, top_n: int = 20):
    feat_imp = pd.DataFrame(
        {
            "feature": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)
    print(f"\nTop {top_n} features by gain:\n{feat_imp.head(top_n).to_string(index=False)}")
    return feat_imp


def save_model(model, feature_cols: list, model_path: str = MODEL_PATH, cols_path: str = FEATURE_COLS_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    with open(cols_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"\nModel saved to {model_path}")
    print(f"Feature columns saved to {cols_path}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)

    # Auto-increment run name based on existing run count
    client = mlflow.tracking.MlflowClient()
    existing_runs = client.search_runs(experiment.experiment_id)
    run_name = f"test-run-{len(existing_runs) + 1}"

    df = load_data()
    X, y, feature_cols = select_features(df)
    X_train, X_val, y_train, y_val = split_data(X, y)

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "num_boost_round": NUM_BOOST_ROUND,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        })

        model = train(X_train, y_train, X_val, y_val)

        # Log validation metrics
        metrics = evaluate(model, X_val, y_val, split_name="Val")
        mlflow.log_metrics({"val_rmse": metrics["rmse"], "val_r2": metrics["r2"]})
        mlflow.log_metric("best_iteration", model.best_iteration)

        feature_importance(model)
        save_model(model, feature_cols)

        # Log model and feature list as artifacts
        mlflow.lightgbm.log_model(model, name="lgbm_model")
        mlflow.log_artifact(FEATURE_COLS_PATH, artifact_path="model_metadata")

        print(f"\nMLflow run logged under experiment: '{EXPERIMENT_NAME}'")
