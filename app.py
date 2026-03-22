import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('heart_model.pkl', 'rb'))

# Page Config - Formal Title
st.set_page_config(page_title="Cardiovascular Diagnostic System", layout="wide")

st.title("Cardiovascular Disease Risk Assessment System")
st.write("This system utilizes a Random Forest Classifier trained on the UCI Cleveland Dataset to predict the probability of heart disease based on clinical biomarkers.")

st.divider()

# --- INPUT SECTION ---
st.subheader("Patient Clinical Data Input")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 1, 100, 50, help="Patient's age.")
    sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest Pain Classification", [0, 1, 2, 3], 
                      help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic.")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x==0 else "True")

with col2:
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], 
                           help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy.")
    thalach = st.slider("Maximum Heart Rate Achieved", 50, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x==1 else "Yes")
    oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3])
    ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (Thal)", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])

st.divider()

# --- PREDICTION ---
if st.button("Generate Diagnostic Report"):
    # Reshape input for model
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    st.subheader("Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"High Risk Detected. Statistical Probability: {prob*100:.2f}%")
        st.write("Clinical Conclusion: The patient's metrics align with patterns of cardiovascular disease. Further diagnostic testing is recommended.")
    else:
        st.success(f"Low Risk Detected. Statistical Probability: {(1-prob)*100:.2f}% (Negative)")
        st.write("Clinical Conclusion: Patient metrics are within statistically normal ranges. Continue routine monitoring.")