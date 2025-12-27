import pandas as pd
from pathlib import Path
from src.data.validate import validate_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Titanic-Dataset.csv"


def load_raw_data():
    """Load raw Titanic dataset."""
    df= pd.read_csv(DATA_PATH)
    validate_schema(df)
    return df


