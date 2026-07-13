import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_xgboost_model.pkl")
metadata = joblib.load("fraud_detection_metadata.pkl")

st.title("Credit Card Fraud Detection System")

st.write(
    "An XGBoost-based machine learning application for "
    "detecting potentially fraudulent transactions."
)

data = None

st.subheader("Select Input Method")

input_method = st.radio(
    "Choose how you want to test the model:",
    ["Try Demo Data", "Upload CSV"]
)

if input_method == "Try Demo Data":
    if st.button("Run Demo"):
        data = pd.read_csv("sample_transactions.csv")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload transaction data",
        type=["csv"]
    )

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

if data is not None:
    feature_columns = metadata["feature_columns"]

    if all(column in data.columns for column in feature_columns):
        input_data = data[feature_columns]

        probabilities = model.predict_proba(input_data)[:, 1]

        predictions = (
            probabilities >= metadata["threshold"]
        ).astype(int)

        results = pd.DataFrame({
            "Transaction Amount": data["TransactionAmt"],
            "Fraud Probability": probabilities,
            "Status": pd.Series(predictions).map(
                {0: "Legitimate", 1: "Fraud"}
            )
        })

        st.subheader("Fraud Detection Results")
        st.dataframe(results)

        fraud_count = int(predictions.sum())
        legitimate_count = len(predictions) - fraud_count

        st.subheader("Prediction Summary")

        col1, col2 = st.columns(2)

        col1.metric(
            "Legitimate Transactions",
            legitimate_count
        )

        col2.metric(
            "Fraudulent Transactions",
            fraud_count
        )

    else:
        st.error(
            "The CSV does not contain the required model features."
        )