import pandas as pd


REQUIRED_COLUMNS = [
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate required columns exist in the dataset."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
