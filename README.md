# Titanic Survival Prediction App

An end-to-end machine learning project that predicts the probability of survival for Titanic passengers using historical data.

The focus of this project is on building a clean, reproducible ML pipeline and a simple Streamlit app for inference.

---

## Project Structure
app/ # Streamlit inference app

data/ # Raw dataset (immutable)

notebooks/ # Exploratory Data Analysis (EDA)

src/ # Production ML code

scripts/ # Utility scripts

artifacts/ # Trained model (ignored)


---

## How to Run

```bash
# create environment
python -m venv .venv
.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# train model
python src/models/train.py

# run app
streamlit run app/streamlit_app.py

