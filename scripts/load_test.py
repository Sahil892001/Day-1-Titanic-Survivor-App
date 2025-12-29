import joblib

model = joblib.load("artifacts/titanic_model.joblib")
print("Model loaded successfully:", type(model))
