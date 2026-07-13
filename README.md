# IEEE-CIS Fraud Detection System

A machine learning project for detecting fraudulent online transactions using the IEEE-CIS Fraud Detection dataset.

The project compares multiple classification models and implements an XGBoost-based fraud detection system with a Streamlit web application.

## Project Overview

Fraud detection is an imbalanced classification problem where fraudulent transactions represent only a small percentage of the dataset.

The IEEE-CIS dataset contains approximately 590,000 transactions with hundreds of anonymized transaction features.

This project focuses on:

- Exploratory Data Analysis
- Missing value handling
- Feature preprocessing
- Class imbalance analysis
- Logistic Regression
- Random Forest
- XGBoost
- Model comparison
- ROC-AUC and PR-AUC evaluation
- Validation-based threshold selection
- Feature importance analysis
- Streamlit application

## Dataset

Dataset: IEEE-CIS Fraud Detection

- Training transactions: 590,540
- Fraud transactions: 20,663
- Fraud percentage: approximately 3.5%

The target variable is:

- `0` - Legitimate Transaction
- `1` - Fraudulent Transaction

## Machine Learning Workflow

1. Data loading and exploration
2. Fraud class distribution analysis
3. Transaction amount analysis
4. Transaction time feature engineering
5. Missing value analysis
6. Train and test data splitting
7. Numerical and categorical preprocessing
8. Categorical feature encoding
9. Model training
10. Model comparison
11. Validation-based threshold selection
12. Final test evaluation
13. Model serialization
14. Streamlit application development

## Models Compared

- Logistic Regression
- Random Forest
- XGBoost

XGBoost achieved the strongest overall performance and was selected as the final model.

## Final Model Performance

| Metric | Score |
|---|---|
| Accuracy | 97.71% |
| Fraud Precision | 70.35% |
| Fraud Recall | 59.59% |
| Fraud F1 Score | 64.53% |
| ROC-AUC | 0.9307 |
| PR-AUC | 0.6794 |
| Classification Threshold | 0.20 |

The classification threshold was selected using validation data based on the fraud-class F1 score.

## Streamlit Application

The Streamlit application allows users to upload transaction data in CSV format.

The application:

- Loads the trained XGBoost model
- Calculates fraud probabilities
- Applies the selected classification threshold
- Classifies transactions as Legitimate or Fraud
- Displays a transaction prediction summary

## Project Structure

```text
ieee-cis-fraud-detection/
│
├── IEEE_CIS_Fraud_Detection.ipynb
├── app.py
├── fraud_detection_xgboost_model.pkl
├── fraud_detection_metadata.pkl
├── sample_transactions.csv
├── fraud_test_samples.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Jupyter Notebook

## Run the Application

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Author

Srijan S I

B.Tech Artificial Intelligence and Data Science