"""Customer Churn Prediction — Logistic Regression + Streamlit."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).with_name("customer_churn.csv")


def create_dataset(rows: int = 1500) -> pd.DataFrame:
    """Create a realistic demo dataset so the project runs without downloads."""
    rng = np.random.default_rng(42)
    tenure = rng.integers(0, 73, rows)
    monthly = rng.normal(70, 25, rows).clip(18, 130).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[.35, .50, .15])
    support = rng.choice(["Yes", "No"], rows, p=[.32, .68])
    paperless = rng.choice(["Yes", "No"], rows, p=[.60, .40])
    score = (-1.2 + 1.25 * (contract == "Month-to-month") + .55 * (internet == "Fiber optic")
             + .65 * (support == "No") + .40 * (paperless == "Yes") - .035 * tenure + .012 * monthly)
    churn = rng.binomial(1, 1 / (1 + np.exp(-score)))
    return pd.DataFrame({"tenure_months": tenure, "monthly_charges": monthly, "contract": contract,
                         "internet_service": internet, "tech_support": support,
                         "paperless_billing": paperless, "churn": churn})


@st.cache_resource
def train_model():
    data = create_dataset()
    X, y = data.drop(columns="churn"), data["churn"]
    categorical = ["contract", "internet_service", "tech_support", "paperless_billing"]
    numerical = ["tenure_months", "monthly_charges"]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {"accuracy": accuracy_score(y_test, predictions), "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
               "report": classification_report(y_test, predictions, output_dict=True)}
    return model, metrics, data


st.set_page_config(page_title="Churn Predictor", page_icon="📉")
model, metrics, data = train_model()
st.title("📉 Customer Churn Predictor")
st.caption("Logistic Regression model trained on a reproducible demo telecom dataset.")

with st.sidebar:
    st.header("Customer details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 18.0, 130.0, 70.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    support = st.selectbox("Technical support", ["Yes", "No"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

customer = pd.DataFrame([{"tenure_months": tenure, "monthly_charges": monthly, "contract": contract,
                          "internet_service": internet, "tech_support": support, "paperless_billing": paperless}])
probability = model.predict_proba(customer)[0, 1]
left, right = st.columns(2)
left.metric("Churn probability", f"{probability:.1%}")
right.metric("Risk level", "High" if probability >= .5 else "Low")

st.subheader("Model evaluation")
a, b = st.columns(2)
a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
b.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
st.info("This dataset is synthetic and intended to demonstrate the complete ML workflow. Replace `create_dataset()` with a real, responsibly sourced dataset for production use.")
