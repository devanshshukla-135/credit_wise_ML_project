import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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

@st.cache_resource
def train_model():
    df = pd.read_csv('loan_approval_data.csv')
    
    # Feature Engineering
    if 'DTI_Ratio' in df.columns:
        df['DTI_Ratio_sq'] = df['DTI_Ratio'] ** 2
    if 'Credit_Score' in df.columns:
        df['Credit_Score_sq'] = df['Credit_Score'] ** 2
    
    education_map = {'Undergraduate': 0, 'Graduate / Higher': 1}
    gender_male_map = {'Female': 0, 'Male': 1}
    marital_single_map = {'Married': 0, 'Single': 1}
    
    # Check original string column names in CSV
    if 'Education_Level' in df.columns:
        df['Education_Level'] = df['Education_Level'].map(education_map)
    if 'Gender' in df.columns:
        df['Gender_Male'] = df['Gender'].map(gender_male_map)
        df.drop(columns=['Gender'], inplace=True)
    elif 'Gender_Male' in df.columns:
        df['Gender_Male'] = df['Gender_Male'].map(gender_male_map)

    if 'Marital_Status' in df.columns:
        df['Marital_Status_Single'] = df['Marital_Status'].map(marital_single_map)
        df.drop(columns=['Marital_Status'], inplace=True)
    elif 'Marital_Status_Single' in df.columns:
        df['Marital_Status_Single'] = df['Marital_Status_Single'].map(marital_single_map)
        
    categorical_cols = ['Employment_Status', 'Employer_Category', 'Property_Area', 'Loan_Purpose']
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=False)
    
    target_col = 'Loan_Approved' if 'Loan_Approved' in df.columns else 'Loan_Status'
    
    X = df.drop(columns=[target_col])
    
    # Ensure all remaining columns are strictly numeric float (Drop leftover text strings if any)
    X = X.select_dtypes(include=[np.number]).astype(float)
    y = df[target_col]
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GaussianNB()
    model.fit(X_scaled, y)
    
    return model, scaler, list(X.columns)

try:
    model, scaler, feature_cols = train_model()
except Exception as e:
    st.error(f"Error initializing training pipeline: {e}")
    st.stop()

# Mappings for UI
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

if st.button("Predict Loan Approval Status", use_container_width=True):
    # Initialize dictionary with all expected features set to 0.0
    row = {feat: 0.0 for feat in feature_cols}

    # Map Numeric & Binary Inputs
    if 'Applicant_Income' in row: row['Applicant_Income'] = float(applicant_income)
    if 'Coapplicant_Income' in row: row['Coapplicant_Income'] = float(coapplicant_income)
    if 'Age' in row: row['Age'] = float(applicant_age)
    if 'Dependents' in row: row['Dependents'] = float(dependents)
    if 'Existing_Loans' in row: row['Existing_Loans'] = float(existing_loans)
    if 'Savings' in row: row['Savings'] = float(savings)
    if 'Collateral_Value' in row: row['Collateral_Value'] = float(collateral_value)
    if 'Loan_Amount' in row: row['Loan_Amount'] = float(loan_amount)
    if 'Loan_Term' in row: row['Loan_Term'] = float(loan_term)
    if 'Education_Level' in row: row['Education_Level'] = float(education_map[education_level])
    if 'DTI_Ratio_sq' in row: row['DTI_Ratio_sq'] = float(dti_ratio ** 2)
    if 'Credit_Score_sq' in row: row['Credit_Score_sq'] = float(credit_score ** 2)
    if 'DTI_Ratio' in row: row['DTI_Ratio'] = float(dti_ratio)
    if 'Credit_Score' in row: row['Credit_Score'] = float(credit_score)
    
    if 'Marital_Status_Single' in row: row['Marital_Status_Single'] = float(marital_single_map[marital_status])
    if 'Gender_Male' in row: row['Gender_Male'] = float(gender_male_map[gender])

    # Map One-Hot Categorical Features
    active_dummies = [
        employment_map.get(employment_status),
        employer_cat_map.get(employer_category),
        property_area_map.get(property_area),
        loan_purpose_map.get(loan_purpose)
    ]
    for dummy in active_dummies:
        if dummy and dummy in row:
            row[dummy] = 1.0

    # Build DataFrame matching exact feature columns sequence
    input_df = pd.DataFrame([row])[feature_cols]
    input_df = input_df.astype(float)

    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        st.subheader("Result:")
        if prediction[0] == 1:
            st.success("🎉 Congratulations! The Loan application is APPROVED.")
        else:
            st.error("❌ We regret to inform you that the Loan application is REJECTED.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
