import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Credit Wise Loan System", layout="centered", page_icon="💳")

st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stNumberInput, .stSelectbox { padding: 2px; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3rem; background-color: #2e6bbf; color: white; }
    .stButton>button:hover { background-color: #1a4a8c; color: white; }
    h1, h2, h3 { color: #31333F; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Wise Loan Approval System")
st.markdown("Enter applicant details below to check loan eligibility.")

# --- 1. Load Models Logic ---
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join('saved_models', 'model.pkl')
        scaler_path = os.path.join('saved_models', 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
    except FileNotFoundError:
        try:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e2:
             st.error(f"Critical Error: Model artifacts missing. Detail: {e2}")
             st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, scaler = load_artifacts()

# --- 2. Feature Mapping Dictionaries ---
education_map = {'Undergraduate': 0, 'Graduate / Higher': 1}
gender_male_map = {'Female': 0, 'Male': 1}
marital_single_map = {'Married': 0, 'Single': 1}

employment_map = {
    'Unemployed': 'Employment_Status_Unemployed',
    'Salaried': 'Employment_Status_Salaried',
    'Self-employed': 'Employment_Status_Self-employed'
}

employer_cat_map = {
    'Unemployed': 'Employer_Category_Unemployed',
    'MNC': 'Employer_Category_MNC',
    'Private': 'Employer_Category_Private',
    'Government': 'Employer_Category_Government'
}

property_area_map = {
    'Urban': 'Property_Area_Urban',
    'Semiurban': 'Property_Area_Semiurban',
    'Rural': 'Property_Area_Rural'
}

loan_purpose_map = {
    'Personal': 'Loan_Purpose_Personal',
    'Education': 'Loan_Purpose_Education',
    'Home': 'Loan_Purpose_Home',
    'Car': 'Loan_Purpose_Car',
    'Other': 'Loan_Purpose_Other'
}

# --- 3. Input Form ---
st.subheader("Personal & Financial Information")
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=50000)
    applicant_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=30)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    savings = st.number_input("Savings", min_value=0, value=20000)
    dependents = st.number_input("Dependents", min_value=0, value=0)
    education_level = st.selectbox("Education Level", ["Graduate / Higher", "Undergraduate"])

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750)
    dti_ratio = st.number_input("DTI Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.3)
    collateral_value = st.number_input("Collateral Value", min_value=0, value=100000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    gender = st.selectbox("Gender", ["Male", "Female"])

st.subheader("Employment & Loan Details")
col3, col4 = st.columns(2)

with col3:
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Private", "MNC", "Government", "Unemployed"])

with col4:
    loan_amount = st.number_input("Loan Amount", min_value=1000, value=50000)
    loan_term = st.number_input("Loan Term (Months)", min_value=6, value=36)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, value=0)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Car", "Education", "Other"])

st.markdown("---")

# --- 4. Prediction Logic ---
if st.button("Predict Loan Approval Status", use_container_width=True):
    # Determine exact feature order from Scaler/Model if available
    if hasattr(scaler, "feature_names_in_"):
        all_features = list(scaler.feature_names_in_)
    else:
        all_features = [
            'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents', 'Existing_Loans', 
            'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term', 'Education_Level',
            'DTI_Ratio_sq', 'Credit_Score_sq', 'Employment_Status_Salaried',
            'Employment_Status_Self-employed', 'Employment_Status_Unemployed',
            'Marital_Status_Single', 'Gender_Male', 'Employer_Category_Government',
            'Employer_Category_MNC', 'Employer_Category_Private',
            'Employer_Category_Unemployed', 'Property_Area_Semiurban',
            'Property_Area_Urban', 'Loan_Purpose_Car', 'Loan_Purpose_Education',
            'Loan_Purpose_Home', 'Loan_Purpose_Personal'
        ]
    
    # Base dictionary setup
    input_data = {col: 0.0 for col in all_features}

    # Derived values
    dti_ratio_sq = float(dti_ratio) ** 2
    credit_score_sq = float(credit_score) ** 2 

    # Populating numeric and binary inputs safely
    mapping_values = {
        'Applicant_Income': float(applicant_income),
        'Age': float(applicant_age),
        'Coapplicant_Income': float(coapplicant_income),
        'Dependents': float(dependents),
        'Existing_Loans': float(existing_loans),
        'Savings': float(savings),
        'Collateral_Value': float(collateral_value),
        'Loan_Amount': float(loan_amount),
        'Loan_Term': float(loan_term),
        'Education_Level': float(education_map[education_level]),
        'DTI_Ratio_sq': dti_ratio_sq,
        'Credit_Score_sq': credit_score_sq,
        'Marital_Status_Single': float(marital_single_map[marital_status]),
        'Gender_Male': float(gender_male_map[gender])
    }

    for k, v in mapping_values.items():
        if k in input_data:
            input_data[k] = v

    # Setting active One-Hot Encoded columns
    one_hot_columns = [
        employment_map[employment_status],
        employer_cat_map[employer_category],
        property_area_map[property_area],
        loan_purpose_map[loan_purpose]
    ]

    for col in one_hot_columns:
        if col in input_data:
            input_data[col] = 1.0

    # Build DataFrame in exact scaler feature order
    input_df = pd.DataFrame([input_data])
    input_df = input_df[all_features]

    try:
        # Scale & Predict cleanly without array manipulation hacks
        scaled_data = scaler.transform(input_df)
        prediction = model.predict(scaled_data)

        st.subheader("Result:")
        if prediction[0] == 1:
            st.success("🎉 Congratulations! The Loan application is APPROVED.")
        else:
            st.error("❌ We regret to inform you that the Loan application is REJECTED.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
