from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_raw_data
from src.preprocessing.pipeline import build_preprocessing_pipeline


def train_model():
    # Load data
    df = load_raw_data()

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build full pipeline
    model = Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_model()
