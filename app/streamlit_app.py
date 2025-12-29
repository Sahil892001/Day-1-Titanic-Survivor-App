import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# --------------------------------------------------
# Title & description
# --------------------------------------------------
st.title("🚢 Titanic Survival Predictor")

st.markdown(
    """
This app estimates the **probability of survival** for a Titanic passenger  
based on historical data and a machine learning model.

⚠️ *This is an educational model, not a historical fact generator.*
"""
)

st.divider()

# --------------------------------------------------
# Load model
# --------------------------------------------------
MODEL_PATH = Path("artifacts") / "titanic_model.joblib"
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Input section
# --------------------------------------------------
st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=30.0)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Survival Probability", use_container_width=True):

    input_df = pd.DataFrame(
        [{
            "Sex": sex,
            "Pclass": pclass,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
        }]
    )

    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="Estimated Survival Probability",
        value=f"{prob:.2%}",
    )

    # Friendly interpretation
    if prob >= 0.7:
        st.success("High likelihood of survival based on the model.")
    elif prob >= 0.4:
        st.warning("Moderate likelihood of survival based on the model.")
    else:
        st.error("Low likelihood of survival based on the model.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption(
    "Built as part of an end-to-end data science project using Python, scikit-learn, and Streamlit."
)