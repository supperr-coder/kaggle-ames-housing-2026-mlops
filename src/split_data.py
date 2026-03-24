"""
One-time script to split the engineered dataset into a train and test file.

Run once before any model development:
    python src/split_data.py

Outputs:
    data/train.csv  — used for all model development (train + val splits)
    data/test.csv   — locked holdout, only touched during final evaluation
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_PATH = "data/AmesHousing_engineered.csv"
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
TEST_SIZE = 0.15
RANDOM_SEED = 42


def main():
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
        print("Split files already exist. Delete them to re-split.")
        return

    df = pd.read_csv(SOURCE_PATH)
    print(f"Loaded: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"train.csv: {train_df.shape}  ->  {TRAIN_PATH}")
    print(f"test.csv:  {test_df.shape}  ->  {TEST_PATH}")


if __name__ == "__main__":
    main()
