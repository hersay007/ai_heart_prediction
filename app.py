import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
try:
    with open('heart_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'heart_model.pkl' not found. Please ensure the model file is in the same directory.")

# Page Configuration - Clinical Theme
st.set_page_config(page_title="Cardiovascular Diagnostic System", layout="wide")

st.title("Cardiovascular Disease Risk Assessment System")
st.write("Diagnostic support tool based on Random Forest Classification and UCI Cleveland clinical data.")

st.divider()

# --- INPUT SECTION ---
st.subheader("Patient Clinical Parameters")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 1, 100, 50, help="उम्र: Patient's age in years.")
    sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female", help="लिंग: Male or Female.")
    cp = st.selectbox("Chest Pain Classification", [0, 1, 2, 3], 
                      help="सीने में दर्द का प्रकार: 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic.")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="आराम के समय रक्तचाप: Normal is usually around 120.")
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, help="कोलेस्ट्रॉल: Fat level in blood.")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x==0 else "True", help="खाली पेट शुगर.")

with col2:
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], help="ईसीजी (ECG) रिपोर्ट.")
    thalach = st.slider("Maximum Heart Rate Achieved", 50, 220, 150, help="दिल की धड़कन की अधिकतम गति.")
    exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", help="व्यायाम के कारण सीने में दर्द.")
    oldpeak = st.slider("ST Depression (Exercise vs Rest)", 0.0, 6.0, 1.0, help="ईसीजी में तनाव का स्तर.")
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3], help="ईसीजी का झुकाव.")
    ca = st.selectbox("Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3], help="मुख्य रक्त वाहिकाओं की संख्या.")
    thal = st.selectbox("Thalassemia Status", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1], help="खून का विकार.")

st.divider()

# --- PREDICTION LOGIC ---
if st.button("Generate Diagnostic Report"):
    # Fix: Define feature names to match training data and avoid UserWarning
    feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    # Create a DataFrame for the prediction
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                             thalach, exang, oldpeak, slope, ca, thal]], 
                            columns=feature_names)
    
    # Perform prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"High Risk Detected. Statistical Probability: {probability*100:.2f}%")
        st.info("Clinical Conclusion: Patient metrics align with cardiovascular disease patterns. Further testing is advised.")
    else:
        st.success(f"Low Risk Detected. Statistical Probability: {(1-probability)*100:.2f}% (Negative)")
        st.info("Clinical Conclusion: Patient parameters are within normal clinical ranges.")