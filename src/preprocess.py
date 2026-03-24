"""
Data loading, feature selection, and train/val splitting.

Expects data/train.csv to exist — run src/split_data.py first.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = "data/train.csv"
TARGET_COL = "Log_SalePrice"
NUNIQUE_THRESHOLD = 5  # columns with < threshold unique values treated as categorical
RANDOM_SEED = 42


def load_data(path: str = TRAIN_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def select_features(df: pd.DataFrame, threshold: int = NUNIQUE_THRESHOLD, feature_cols: list = None):
    """Split target off and return only numerical feature columns.

    Args:
        df: DataFrame containing features and target.
        threshold: Minimum unique values to treat a column as numerical.
                   Only used when feature_cols is None (i.e. on training data).
        feature_cols: Explicit list of columns to select. Pass the list saved
                      from training to guarantee consistent features at eval time.
    """
    y = df[TARGET_COL]
    df = df.drop(columns=[TARGET_COL])

    if feature_cols is None:
        feature_cols = [col for col in df.columns if df[col].nunique() >= threshold]
        print(f"Features selected from data: {len(feature_cols)} numerical columns (nunique >= {threshold})")
    else:
        print(f"Features applied from saved list: {len(feature_cols)} columns")

    X = df[feature_cols]
    print(f"Target shape: {y.shape}")
    return X, y, feature_cols


def split_data(X: pd.DataFrame, y: pd.Series, random_seed: int = RANDOM_SEED):
    """80% train / 20% val split on the pre-split train file."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, shuffle=True
    )
    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}")
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    df = load_data()
    X, y, feature_cols = select_features(df)
    X_train, X_val, y_train, y_val = split_data(X, y)
