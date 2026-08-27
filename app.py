import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Wise Loan System", layout="wide")

st.title("Credit Wise Loan Approval System")
st.write("Enter applicant details below to check loan eligibility.")

@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('features.pkl', 'rb') as f:
        expected_features = pickle.load(f)
    return model, scaler, expected_features

try:
    model, scaler, expected_features = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    education_level = st.selectbox("Education Level", ["Undergraduate", "Graduate", "Postgraduate"])

with col2:
    income = st.number_input("Applicant Income", min_value=0.0, value=50000.0)
    savings = st.number_input("Savings", min_value=0.0, value=10000.0)
    collateral = st.number_input("Collateral Value", min_value=0.0, value=0.0)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Government", "Private", "Business", "Unemployed"])

with col3:
    loan_amount = st.number_input("Loan Amount Requested", min_value=0.0, value=100000.0)
    loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, value=0)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Home", "Personal"])

col_sub1, col_sub2 = st.columns(2)
with col_sub1:
    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=750.0)
with col_sub2:
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.30)

if st.button("Predict Loan Status"):
    raw_data = {
        'Age': age,
        'Income': income,
        'Savings': savings,
        'Collateral_Value': collateral,
        'Loan_Amount': loan_amount,
        'Loan_Term': loan_term,
        'Existing_Loans': existing_loans,
        'Credit_Score': credit_score,
        'DTI_Ratio': dti_ratio,
        'Dependents': dependents,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Marital_Status_Married': 1 if marital_status == "Married" else 0,
        'Marital_Status_Single': 1 if marital_status == "Single" else 0,
        'Education_Graduate': 1 if education_level == "Graduate" else 0,
        'Education_Postgraduate': 1 if education_level == "Postgraduate" else 0,
        'Education_Undergraduate': 1 if education_level == "Undergraduate" else 0,
        'Employment_Status_Salaried': 1 if employment_status == "Salaried" else 0,
        'Employment_Status_Self-Employed': 1 if employment_status == "Self-Employed" else 0,
        'Employment_Status_Unemployed': 1 if employment_status == "Unemployed" else 0,
        'Employer_Category_Government': 1 if employer_category == "Government" else 0,
        'Employer_Category_Private': 1 if employer_category == "Private" else 0,
        'Employer_Category_Business': 1 if employer_category == "Business" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0,
        'Property_Area_Rural': 1 if property_area == "Rural" else 0,
        'Loan_Purpose_Car': 1 if loan_purpose == "Car" else 0,
        'Loan_Purpose_Education': 1 if loan_purpose == "Education" else 0,
        'Loan_Purpose_Home': 1 if loan_purpose == "Home" else 0,
        'Loan_Purpose_Personal': 1 if loan_purpose == "Personal" else 0,
        'DTI_Ratio_sq': dti_ratio ** 2,
        'Credit_Score_sq': credit_score ** 2
    }

    input_df = pd.DataFrame([raw_data])

    if expected_features:
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

    scaled_array = scaler.transform(input_df)

    target_n = model.n_features_in_
    if scaled_array.shape[1] < target_n:
        scaled_array = np.pad(scaled_array, ((0, 0), (0, target_n - scaled_array.shape[1])), 'constant')
    elif scaled_array.shape[1] > target_n:
        scaled_array = scaled_array[:, :target_n]

    prediction = model.predict(scaled_array)

    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("🎉 Congratulations! The Loan application is APPROVED.")
    else:
        st.error("❌ Sorry, The Loan application is REJECTED.")
