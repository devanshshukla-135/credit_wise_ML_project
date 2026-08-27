import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Wise Loan System",
    page_icon="🏦",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, scaler, features


try:
    model, scaler, features = load_files()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()


# ---------------- TITLE ----------------
st.title("🏦 Credit Wise Loan System")
st.write("Enter applicant details to predict loan status.")

st.divider()


# ---------------- INPUTS ----------------
st.subheader("Applicant Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

employment_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    value=5.0
)

interest_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.20
)

credit_history_length = st.number_input(
    "Credit History Length",
    min_value=0.0,
    value=5.0
)

home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

loan_intent = st.selectbox(
    "Loan Purpose",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

previous_default = st.selectbox(
    "Previous Default",
    ["N", "Y"]
)


# ---------------- PREDICTION ----------------
if st.button("Predict Loan Status"):

    # Create input dictionary
    input_data = {
        "person_age": age,
        "person_income": income,
        "person_emp_length": employment_length,
        "loan_amnt": loan_amount,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": credit_history_length,

        "person_home_ownership_RENT": 1 if home_ownership == "RENT" else 0,
        "person_home_ownership_OWN": 1 if home_ownership == "OWN" else 0,
        "person_home_ownership_MORTGAGE": 1 if home_ownership == "MORTGAGE" else 0,
        "person_home_ownership_OTHER": 1 if home_ownership == "OTHER" else 0,

        "loan_intent_PERSONAL": 1 if loan_intent == "PERSONAL" else 0,
        "loan_intent_EDUCATION": 1 if loan_intent == "EDUCATION" else 0,
        "loan_intent_MEDICAL": 1 if loan_intent == "MEDICAL" else 0,
        "loan_intent_VENTURE": 1 if loan_intent == "VENTURE" else 0,
        "loan_intent_HOMEIMPROVEMENT": 1 if loan_intent == "HOMEIMPROVEMENT" else 0,
        "loan_intent_DEBTCONSOLIDATION": 1 if loan_intent == "DEBTCONSOLIDATION" else 0,

        "loan_grade_A": 1 if loan_grade == "A" else 0,
        "loan_grade_B": 1 if loan_grade == "B" else 0,
        "loan_grade_C": 1 if loan_grade == "C" else 0,
        "loan_grade_D": 1 if loan_grade == "D" else 0,
        "loan_grade_E": 1 if loan_grade == "E" else 0,
        "loan_grade_F": 1 if loan_grade == "F" else 0,
        "loan_grade_G": 1 if loan_grade == "G" else 0,

        "cb_person_default_on_file_N": 1 if previous_default == "N" else 0,
        "cb_person_default_on_file_Y": 1 if previous_default == "Y" else 0,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # IMPORTANT:
    # Exactly match the features used while training
    input_df = input_df.reindex(columns=features, fill_value=0)

    try:
        # Scale the data
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)

        st.divider()

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")

    except Exception as e:
        st.error("Prediction Error")
        st.write(e)

        st.write("Expected Features:", len(features))
        st.write("Input Features:", input_df.shape[1])
