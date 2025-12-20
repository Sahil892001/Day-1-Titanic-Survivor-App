import pandas as pd
from pathlib import Path
from src.data.validate import validate_schema

DATA_PATH = Path("data/raw/Titanic-Dataset.csv")


def load_raw_data():
    """Load raw Titanic dataset."""
    df= pd.read_csv(DATA_PATH)
    validate_schema(df)
    return df


