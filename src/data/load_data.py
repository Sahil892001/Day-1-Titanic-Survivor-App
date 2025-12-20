import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/Titanic-Dataset.csv")


def load_raw_data():
    """Load raw Titanic dataset."""
    return pd.read_csv(DATA_PATH)


