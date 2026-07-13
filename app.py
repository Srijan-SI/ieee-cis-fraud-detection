import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_xgboost_model.pkl")
metadata = joblib.load("fraud_detection_metadata.pkl")

st.title("Credit Card Fraud Detection System")

st.write(
    "This application uses an XGBoost machine learning model "
    "to classify transactions as legitimate or fraudulent."
)

uploaded_file = st.file_uploader(
    "Upload transaction data",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transaction Data")

    display_columns = [
        column
        for column in ["TransactionDT", "TransactionAmt"]
        if column in data.columns
    ]

    st.dataframe(data[display_columns].head(10))

    feature_columns = metadata["feature_columns"]

    if all(column in data.columns for column in feature_columns):
        input_data = data[feature_columns]

        probabilities = model.predict_proba(input_data)[:, 1]

        predictions = (
            probabilities >= metadata["threshold"]
        ).astype(int)

        results = pd.DataFrame()

        if "TransactionDT" in data.columns:
            results["TransactionDT"] = data["TransactionDT"]

        if "TransactionAmt" in data.columns:
            results["TransactionAmt"] = data["TransactionAmt"]

        results["Fraud Probability"] = probabilities
        results["Status"] = pd.Series(predictions).map(
            {0: "Legitimate", 1: "Fraud"}
        )

        st.subheader("Fraud Detection Results")
        st.dataframe(results)

        fraud_count = int(predictions.sum())
        legitimate_count = int(len(predictions) - fraud_count)

        st.subheader("Prediction Summary")

        col1, col2 = st.columns(2)

        col1.metric("Legitimate Transactions", legitimate_count)
        col2.metric("Fraudulent Transactions", fraud_count)

    else:
        st.error(
            "Uploaded CSV does not contain the required model features."
        )