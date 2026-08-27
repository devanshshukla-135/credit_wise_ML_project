import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Credit Wise Loan System")

# 1. Model aur Scaler Load Karein
# Important: Agar scaling training mein use ki thi, toh scaler.pkl load karna must hai.
@st.cache_resource
def load_artifacts():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        # Agar scaling lagayi thi training mein, toh 'scaler.pkl' load karo.
        # scaler = pickle.load(open("scaler.pkl", "rb"))
        return model # , scaler (agar scaler use kiya ho toh uncomment karein)
    except FileNotFoundError:
        st.error("Error: 'model.pkl' (ya scaler.pkl) nahi mila. Please files upload karein.")
        st.stop()

model = load_artifacts()

# 2. Form Inputs (Make sure unique values exact training data jaisi hon)
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
    # manual mapping dictionaries (training data logic ke hisab se)
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    dependents_map = {'0': 0, '1': 1, '2': 2, '3+': 3}

    # input data array taiyar karo (EXACT feature order mein jo training mein tha)
    # GaussianNB direct manual encoded values accept karta hai agar scalar used na ho.
    raw_data = np.array([[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        gender_map[gender],
        married_map[married],
        education_map[education],
        self_employed_map[self_employed],
        property_area_map[property_area],
        dependents_map[dependents]
    ]])

    try:
        # Step: Agar StandardScaler used tha, yahan transform apply karein:
        # raw_data = scaler.transform(raw_data) 

        # Prediction
        prediction = model.predict(raw_data)[0]
        
        # Binary Classification Check (1/0)
        if prediction == 1:
            st.success("🎉 Congratulations! Loan Approved.")
        else:
            st.error("❌ Sorry! Loan Rejected.")

    except Exception as e:
        # Detailed error reporting feature count ke liye
        if "X has" in str(e):
             st.error(f"Error: Feature Mismatch. Model expected 27 features, but received {raw_data.shape[1]}. **This requires model retraining with numerical features only.**")
        else:
             st.error(f"Prediction Error: {e}")

