# Customer Churn Prediction

A portfolio-ready binary-classification project that predicts whether a telecom customer is likely to churn, using **Logistic Regression**.

## Highlights
- Reproducible synthetic dataset with customer tenure, billing, contract, internet, and support features
- `ColumnTransformer` pipeline for numerical scaling and categorical one-hot encoding
- Stratified train/test split; test accuracy and ROC-AUC metrics
- Streamlit interface for real-time churn-risk predictions

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Resume bullet
Built an end-to-end telecom churn prediction application using Logistic Regression, scikit-learn preprocessing pipelines, and Streamlit; evaluated model performance with accuracy and ROC-AUC on a stratified holdout dataset.
