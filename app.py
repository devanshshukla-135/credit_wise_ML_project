import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Page Config & Styling ---
st.set_page_config(page_title="Credit Wise Loan System", layout="centered", page_icon="💳")

# Custom CSS for design to make phone usage slightly easier
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
        # Standard paths, assuming files are in 'saved_models' directory for clean deployment
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
        # Fallback agar saved_models folder na mile (agar app structure different hai)
        try:
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            return model, scaler
        except Exception as e2:
             st.error(f"Critical Error: Model artifacts ('model.pkl' or 'scaler.pkl') missing. Check file uploads or 'saved_models' directory. Detail: {e2}")
             st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Auto-execute on load
model, scaler = load_artifacts()

# --- 2. Feature Mapping Dictionaries ---
education_map = {'Undergraduate': 0, 'Graduate / Higher': 1}
# Standard Binary mapping based on training data
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
    # ADDED 'Age' input here to fix the 26 vs 27 feature count mismatch
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
    
    # 4.1 Base DataFrame with Zeros to ensure exactly 27 initial columns
    input_df = pd.DataFrame(columns=all_features)
    input_df.loc[0] = [0.0] * len(all_features) # Using floats by default is safer

    # 4.2 Feature transformations (Matching Notebook logic)
    dti_ratio_sq = dti_ratio ** 2
    # Important: Cast to float before squaring for numerical stability
    credit_score_sq = float(credit_score) ** 2 

    # 4.3 Fill Numerical & Transformed features (Including the missed 'Age' feature)
    # The 'Age' column in notebook needs data from form or explicit fill
    input_df['Age'] = float(applicant_age) # Now using data from the form input
    input_df['Coapplicant_Income'] = float(coapplicant_income)
    input_df['Dependents'] = float(dependents) # Using float as safest representation
    input_df['Existing_Loans'] = float(existing_loans)
    input_df['Savings'] = float(savings)
    input_df['Collateral_Value'] = float(collateral_value)
    input_df['Loan_Amount'] = float(loan_amount)
    input_df['Loan_Term'] = float(loan_term)
    input_df['Education_Level'] = float(education_map[education_level])
    input_df['DTI_Ratio_sq'] = dti_ratio_sq
    input_df['Credit_Score_sq'] = credit_score_sq

    # 4.4 Fill Binary Features (Mappings to float)
    input_df['Marital_Status_Single'] = float(marital_single_map[marital_status])
    input_df['Gender_Male'] = float(gender_male_map[gender])

    # 4.5 Fill One-Hot Encoded Dummies Replication Logic
    # These columns exist in input_df due to initialization with 'all_features'
    one_hot_columns_to_fill = [
        employment_map[employment_status],
        employer_cat_map[employer_category],
        property_area_map[property_area],
        loan_purpose_map[loan_purpose]
    ]

    # Set 1.0 only for the specific active dummy column
    for col in one_hot_columns_to_fill:
        if col in input_df.columns:
            input_df[col] = 1.0

    # Verification Step: Now input_df must have exactly 27 features.
    # (Checking shape before realignment to prevent alignment errors)
    current_feature_count = input_df.shape[1]
    if current_feature_count != 27:
        st.error(f"Internal Feature Mismatch Error: Final input dataframe has {current_feature_count} features. Expected exactly 27. Re-check code structure.")
        # Verifying which features are present (debug only)
        # st.write("Current Columns:", list(input_df.columns))
        st.stop()

    # 5. --- CRITICAL RE-ALIGNMENT FOR STRICT SCALER VALIDATION ---
    # The order of columns must be exact as they were during fit().
    try:
        # Step A: Auto-detect Scaler's training order (Preferred method)
        if hasattr(scaler, "feature_names_in_"):
            expected_scaler_order = list(scaler.feature_names_in_)
        else:
            # Plan B: Backup manual list backup use karein
            st.warning("Warning: Could not auto-detect scaler feature names in strict mode. Falling back to manual order.")
            expected_scaler_order = all_features # Assuming manual list was reliable reference during training

        # Step B: Strict realignment (Columns re-order + ensure exact match count)
        aligned_input_df = input_df.reindex(columns=expected_scaler_order)
        
        # Double check alignment visually (debug)
        # st.write("Aligned DataFrame Column names before scaling:", list(aligned_input_df.columns))

        # Step C: Scale the strict-aligned DataFrame
        # StandardScaler expects all features to exist and be aligned
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
        # Improved error pinpointing - likely issues with alignment logic, model mismatch, or scaling logic
        st.error(f"Execution Error: An unexpected issue occurred during feature alignment, scaling, or prediction. Ensure model.pkl and scaler.pkl match input format. Detail: {e}")
        
