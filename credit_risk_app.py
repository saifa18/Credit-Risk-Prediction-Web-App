import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. LOAD THE PRE-TRAINED MODEL
MODEL_PATH = "credit_risk_model.pkl"

def load_model():
   model = joblib.load(MODEL_PATH)
   return model

# 2. STREAMLIT APP
def main():
    st.title("Minimal Credit Risk Predictor")
    model = load_model()
    st.write("Enter the following features to get a default probability:")

# 3. COLLECT USER INPUT

    loan_amnt = st.number_input(
        "Enter the Loan Amount (e.g., 5000)",
        min_value=0,
        value=5000,
        step=100
    )
    annual_income = st.number_input(
        "Enter the Annual Income (e.g., 50000)",
        min_value=0,
        value=50000,
        step=1000
    )

    loan_percent_income = st.slider("Loan-to-Income Ratio", 0.0, 1.0, 0.2, 0.01)
    loan_int_rate       = st.slider("Interest Rate (%)", 5.0, 25.0, 10.0, 0.1)

    # Selectbox or checkbox for home ownership
    rent_val = st.selectbox("Home Ownership: Rent (1=Yes, 0=No)", [0, 1])  # 0 = no, 1 = yes
    own_val  = st.selectbox("Home Ownership: Own (1=Yes, 0=No)",  [0, 1])  # 0 = no, 1 = yes

    cb_val   = st.selectbox("Previous Credit Default? (1=Yes, 0=No)", [0, 1]) # 0 = no, 1 = yes
    person_emp_length  = st.number_input("Employment Length (Years", min_value=0, max_value=50, value=10)
    loan_amnt_log = np.log1p(loan_amnt)        # log(loan_amnt + 1)
    person_income_log = np.log1p(annual_income)
# 4. GENERATE PREDICTION WHEN BUTTON IS CLICKED

    if st.button("Predict Default Probability"):
        # Create a 1-row dataframe that matches the training column order
        input_data = {
            "loan_percent_income": [loan_percent_income],
            "loan_int_rate": [loan_int_rate],
            "person_home_ownership_RENT": [rent_val],
            "cb_person_default_on_file_Y": [cb_val],
            "loan_amnt_log": [loan_amnt_log],
            "person_income_log": [person_income_log],
            "person_home_ownership_OWN": [own_val],
            "person_emp_length": [person_emp_length]
        }
        
        # Predict default probability (the second column of predict_proba)
        df_input = pd.DataFrame(input_data)
        pred_proba = model.predict_proba(df_input)[0][1]   
        st.write(f"**Estimated Default Probability:** {pred_proba:.2f}")

# 5. RUN APP
if __name__ == "__main__":
    main()
