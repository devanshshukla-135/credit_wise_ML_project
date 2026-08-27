import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Credit Wise Loan System", layout="centered", page_icon="💳")

# Custom CSS for design
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

# --- 1. Robust Load Models Logic ---
@st.cache_resource
def load_artifacts():
    try:
        # Streamlit deployment (mount/src/...) ke liye correct path
        # Agar local mein chalayein, toh ye folders exist karne chahiye
        model_path = os.path.join('saved_models', 'model.pkl')
        scaler_path = os.path.join('saved_models', 'scaler.pkl')
        
        # Load Model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Load Scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
    except FileNotFoundError as e:
        # Fallback agar saved_models folder deployment mein na mile (local dev only)
        # Agar production mein ye error aaye, toh deployment structure check karein
        try:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e2:
             st.error(f"Critical Error: Model artifacts ('saved_models/model.pkl' or 'saved_models/scaler.pkl') missing. Check file uploads. Detail: {e2}")
             st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Auto-execute on load
model, scaler = load_artifacts()

# --- 2. Feature Mapping Dictionaries ---
education_map = {'Undergraduate': 0, 'Graduate / Higher': 1}
# Standard Binary mapping
gender_male_map = {'Female': 0, 'Male': 1}
marital_single_map = {'Married': 0, 'Single': 1}

# Dictionaries for One-Hot Encoding replicate logic
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
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    savings = st.number_input("Savings", min_value=0, value=20000)
    collateral_value = st.number_input("Collateral Value", min_value=0, value=100000)
    dependents = st.number_input("Dependents", min_value=0, value=0)
    education_level = st.selectbox("Education Level", ["Graduate / Higher", "Undergraduate"])

with col2:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750)
    dti_ratio = st.number_input("DTI Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.3)
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

# --- 4. Prediction Logic (Click handler) ---
if st.button("Predict Loan Approval Status", use_container_width=True):
    # CRITICAL: Copy of ALL notebook training feature names in EXACT order for DataFrame init
    all_features = [
        'Coapplicant_Income', 'Age', 'Dependents', 'Existing_Loans', 'Savings',
        'Collateral_Value', 'Loan_Amount', 'Loan_Term', 'Education_Level',
        'DTI_Ratio_sq', 'Credit_Score_sq', 'Employment_Status_Salaried',
        'Employment_Status_Self-employed', 'Employment_Status_Unemployed',
        'Marital_Status_Single', 'Gender_Male', 'Employer_Category_Government',
        'Employer_Category_MNC', 'Employer_Category_Private',
        'Employer_Category_Unemployed', 'Property_Area_Semiurban',
        'Property_Area_Urban', 'Loan_Purpose_Car', 'Loan_Purpose_Education',
        'Loan_Purpose_Home', 'Loan_Purpose_Personal'
    ]
    
    # 4.1 Base DataFrame with Zeros
    input_df = pd.DataFrame(columns=all_features)
    input_df.loc[0] = [0] * len(all_features)

    # 4.2 Feature transformations (Matching Notebook logic)
    dti_ratio_sq = dti_ratio ** 2
    credit_score_sq = float(credit_score) ** 2  # Convert to float before squaring

    # 4.3 Fill Numerical & Scaled-to-be features
    # (Pichhle code se steps 3, 4, 5 ke mapping fills fused yahan)
    input_df['Age'] = 30 # Default for stable training if Age missing from form
    input_df['Coapplicant_Income'] = float(coapplicant_income)
    input_df['Dependents'] = int(dependents)
    input_df['Existing_Loans'] = int(existing_loans)
    input_df['Savings'] = float(savings)
    input_df['Collateral_Value'] = float(collateral_value)
    input_df['Loan_Amount'] = float(loan_amount)
    input_df['Loan_Term'] = int(loan_term)
    input_df['Education_Level'] = education_map[education_level]
    input_df['DTI_Ratio_sq'] = dti_ratio_sq
    input_df['Credit_Score_sq'] = credit_score_sq

    # 4.4 Fill Binary Features
    input_df['Marital_Status_Single'] = marital_single_map[marital_status]
    input_df['Gender_Male'] = gender_male_map[gender]

    # 4.5 Fill One-Hot Encoded Dummies (Only if that column exists)
    one_hot_columns_to_fill = [
        employment_map[employment_status],
        employer_cat_map[employer_category],
        property_area_map[property_area],
        loan_purpose_map[loan_purpose]
    ]

    for col in one_hot_columns_to_fill:
        if col in input_df.columns:
            input_df[col] = 1

    # Verify input_df before alignment (debug)
    # st.write("Raw input dataframe:", input_df)

    # 5. --- CRITICAL RE-ALIGNMENT FOR STRICT SCALER VALIDATION ---
    try:
        # Step A: Auto-detect Scaler's training order
        if hasattr(scaler, "feature_names_in_"):
            expected_scaler_order = list(scaler.feature_names_in_)
        else:
            # Plan B: Backup manual list backup use karein
            expected_scaler_order = all_features # Assuming manual list was reliable training reference
            # st.warning("Autodetect failed, using manual order for scaler. Mismatch potential.") # debug only

        # Step B: Strict realignment (Columns re-order + ensure no mismatch count)
        aligned_input_df = input_df.reindex(columns=expected_scaler_order)
        
        # Verify alignment visually before scaling (debug)
        # st.write("Aligned DataFrame Column names before scaling:", list(aligned_input_df.columns))

        # Step C: Scale the strict-aligned DataFrame
        scaled_data = scaler.transform(aligned_input_df)
        
        # Step D: Predict
        prediction = model.predict(scaled_data)

        # Step E: Output Condition (Assuming based on model fit, 1=Approved, 0=Rejected)
        st.subheader("Result:")
        if prediction[0] == 1:
            st.success("🎉 Congratulations! The Loan application is APPROVED.")
        else:
            st.error("❌ We regret to inform you that the Loan application is REJECTED.")
            
    except Exception as e:
        # pinpointing the issue now, likely either column alignment logic error or model training mismatch
        st.error(f"Error: An unexpected issue occurred during prediction or alignment. Detail: {e}")
        
    
