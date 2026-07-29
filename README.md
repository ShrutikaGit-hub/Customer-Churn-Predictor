1. Customer Churn Prediction — README.md
# Customer Churn Prediction using Logistic Regression

This project predicts whether a telecom customer is likely to leave the service. It uses a Logistic Regression machine learning model and provides an interactive Streamlit web application for churn prediction.

## Objective

The objective of this project is to identify customers who may churn based on their account and service information. This can help businesses take preventive actions, such as offering better plans or customer support.

## Algorithm Used

- Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for binary classification. In this project, it predicts one of two outcomes:

- `1` — Customer is likely to churn
- `0` — Customer is not likely to churn

## Features Used

- Customer tenure in months
- Monthly charges
- Contract type
- Internet service type
- Technical support availability
- Paperless billing preference

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## ML Workflow

1. Generate/load customer data.
2. Separate features and churn target.
3. Preprocess numerical data using `StandardScaler`.
4. Convert categorical columns using one-hot encoding.
5. Split data into training and testing sets.
6. Train a Logistic Regression model.
7. Evaluate the model using Accuracy and ROC-AUC.
8. Predict churn probability through the Streamlit interface.

## Project Snap

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/124d7103-164e-41f9-a62e-9015cffbc767" />
