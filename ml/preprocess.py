import os

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "creditcard.csv.gz")


def load_data(path=DATA_PATH):
   
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md."
        )

    df = pd.read_csv(path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
