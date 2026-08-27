
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load saved model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Credit Wise Loan System", layout="wide")

st.title("💳 Credit Wise Loan Approval System")
st.write("Fill in the applicant details below to check loan eligibility.")

# Layout in Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    education_level = st.selectbox("Education Level", ["Undergraduate", "Graduate / Higher"])

with col2:
    st.subheader("Financial & Employment")
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    savings = st.number_input("Savings", min_value=0.0, value=50000.0)
    collateral_value = st.number_input("Collateral Value", min_value=0.0, value=100000.0)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Government", "MNC", "Private", "Unemployed"])

with col3:
    st.subheader("Loan Details")
    loan_amount = st.number_input("Loan Amount", min_value=1000.0, value=200000.0)
    loan_term = st.number_input("Loan Term (Months)", min_value=6, max_value=360, value=36)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=0)
    property_area = st.selectbox("Property Area", ["Semiurban", "Urban", "Rural"])
    loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal", "Other"])
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=750.0)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)

st.markdown("---")

if st.button("Predict Loan Status", use_container_width=True):
    # Engineered Features
    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = credit_score ** 2

    # Categorical One-Hot Conversions
    education_val = 1 if education_level == "Graduate / Higher" else 0
    emp_salaried = 1 if employment_status == "Salaried" else 0
    emp_self = 1 if employment_status == "Self-employed" else 0
    emp_unemployed = 1 if employment_status == "Unemployed" else 0

    marital_single = 1 if marital_status == "Single" else 0
    gender_male = 1 if gender == "Male" else 0

    emp_cat_gov = 1 if employer_category == "Government" else 0
    emp_cat_mnc = 1 if employer_category == "MNC" else 0
    emp_cat_pvt = 1 if employer_category == "Private" else 0
    emp_cat_unemp = 1 if employer_category == "Unemployed" else 0

    prop_semiurban = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0

    purpose_car = 1 if loan_purpose == "Car" else 0
    purpose_edu = 1 if loan_purpose == "Education" else 0
    purpose_home = 1 if loan_purpose == "Home" else 0
    purpose_personal = 1 if loan_purpose == "Personal" else 0

    # Build DataFrame matching exact feature list and column order
    input_df = pd.DataFrame([{
        'Coapplicant_Income': coapplicant_income,
        'Age': age,
        'Dependents': dependents,
        'Existing_Loans': existing_loans,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Education_Level': education_val,
        'Employment_Status_Salaried': emp_salaried,
        'Employment_Status_Self-employed': emp_self,
        'Employment_Status_Unemployed': emp_unemployed,
        'Marital_Status_Single': marital_single,
        'Gender_Male': gender_male,
        'Employer_Category_Government': emp_cat_gov,
        'Employer_Category_MNC': emp_cat_mnc,
        'Employer_Category_Private': emp_cat_pvt,
        'Employer_Category_Unemployed': emp_cat_unemp,
        'Property_Area_Semiurban': prop_semiurban,
        'Property_Area_Urban': prop_urban,
        'Loan_Purpose_Car': purpose_car,
        'Loan_Purpose_Education': purpose_edu,
        'Loan_Purpose_Home': purpose_home,
        'Loan_Purpose_Personal': purpose_personal,
        'DTI_Ratio_sq': dti_ratio_sq,
        'Credit_Score_sq': credit_score_sq
    }])

    # Transform & Predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("🎉 Congratulations! The Loan application is APPROVED.")
    else:
        st.error("❌ We regret to inform you that the Loan application is REJECTED.")