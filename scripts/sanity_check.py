from src.data.load_data import load_raw_data
from src.preprocessing.pipeline import build_preprocessing_pipeline

df = load_raw_data()
X = df.drop(columns=["Survived"])

preprocessor = build_preprocessing_pipeline()
X_transformed = preprocessor.fit_transform(X)

print(X_transformed.shape)