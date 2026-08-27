import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Credit Wise Loan System")

# 1. Model aur Scaler/Encoder load karein
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_artifacts()

# 2. Form Inputs (Apne app ke hisab se fields adjust karein)
st.subheader("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_term = st.number_input("Loan Amount Term", min_value=0, value=360)

with col2:
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

# 3. Prediction Logic
if st.button("Predict Loan Status"):
    # Raw input data dictionary
    raw_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Gender': gender,
        'Married': married,
        'Education': education,
        'Self_Employed': self_employed,
        'Property_Area': property_area,
        'Dependents': dependents
    }

    df_input = pd.DataFrame([raw_data])

    # Category encoding (Training ke exact logic ke saath match karein)
    df_encoded = pd.get_dummies(df_input)

    # -------------------------------------------------------------
    # CRITICAL FIX: Align features with model's expected inputs (27 features)
    # -------------------------------------------------------------
    if hasattr(model, "feature_names_in_"):
        expected_cols = model.feature_names_in_
        # Reindex missing columns ko 0 se fill karega aur exact order set karega
        df_encoded = df_encoded.reindex(columns=expected_cols, fill_value=0)
    
    try:
        # Prediction
        prediction = model.predict(df_encoded)[0]
        
        # Binary Classification Check (1/0 ya 'Y'/'N')
        if prediction == 1 or str(prediction).upper() == 'Y':
            st.success("🎉 Congratulations! Loan Approved.")
        else:
            st.error("❌ Sorry! Loan Rejected.")

        # Agar probability threshold check karni ho:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_encoded)[0]
            st.info(f"Approval Probability: {proba[1]*100:.2f}% | Rejection Probability: {proba[0]*100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
