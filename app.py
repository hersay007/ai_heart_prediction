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

# Page Configuration
st.set_page_config(page_title="Cardiovascular Diagnostic System", layout="wide")

st.title("Cardiovascular Disease Risk Assessment System")
st.write("Clinical decision support tool using Machine Learning for heart health analysis.")

st.divider()

# --- INPUT SECTION ---
st.subheader("Patient Clinical Parameters")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 1, 100, 50, 
                    help="Patient's age. | उम्र: मरीज की आयु।")
    
    sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female", 
                       help="Male or Female. | लिंग: पुरुष या महिला।")
    
    cp = st.selectbox("Chest Pain Classification", [0, 1, 2, 3], 
                      help="Type of chest pain. 0: Severe, 1: Moderate, 2: Non-heart related, 3: No pain. | सीने में दर्द का प्रकार: 0: गंभीर, 1: सामान्य, 2: हृदय से संबंधित नहीं, 3: कोई लक्षण नहीं।")
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, 
                               help="Blood pressure while resting. Normal is ~120. | रक्तचाप: आराम के समय खून का दबाव (सामान्य 120 होता है)।")
    
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200, 
                           help="Amount of fat in blood. High is >240. | कोलेस्ट्रॉल: खून में चर्बी की मात्रा।")
    
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x==0 else "True", 
                       help="Sugar level after fasting. True indicates diabetes risk. | खाली पेट शुगर: क्या शुगर 120 से ज्यादा है? (हाँ मतलब मधुमेह का खतरा)।")

with col2:
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2], 
                           help="ECG results at rest. 0: Normal, 1/2: Abnormalities. | ईसीजी (ECG) रिपोर्ट: 0: सामान्य, 1/2: दिल की धड़कन में असामान्यता।")
    
    thalach = st.slider("Maximum Heart Rate Achieved", 50, 220, 150, 
                        help="Highest heart rate during exercise. | अधिकतम धड़कन: व्यायाम के दौरान दिल की सबसे तेज़ गति।")
    
    exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x==1 else "No", 
                         help="Chest pain while walking or running. | व्यायाम के दौरान दर्द: क्या चलने या दौड़ने पर सीने में दर्द होता है?")
    
    oldpeak = st.slider("ST Depression (Heart Stress)", 0.0, 6.0, 1.0, 
                        help="Stress level of the heart during activity. | दिल पर तनाव: काम के दौरान दिल पर पड़ने वाला दबाव।")
    
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2, 3], 
                         help="Shape of heart wave during peak activity. | ईसीजी लहर का झुकाव: हृदय की लहरों का ग्राफ।")
    
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3], 
                      help="Number of clear blood pipes. 0 is best. | बंद नसें: मुख्य रक्त वाहिकाओं की संख्या (0 सबसे अच्छा है)।")
    
    thal = st.selectbox("Thalassemia / Blood Flow", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1], 
                        help="Blood flow status. 1: Normal, 2/3: Issues. | रक्त प्रवाह की स्थिति: दिल तक खून पहुँचने की जांच।")

st.divider()

# --- PREDICTION LOGIC ---
if st.button("Generate Diagnostic Report"):
    # Fix for UserWarning: Use DataFrame with feature names
    feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                             thalach, exang, oldpeak, slope, ca, thal]], 
                            columns=feature_names)
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Statistical Analysis")
    
    if prediction[0] == 1:
        st.error(f"High Risk Detected. Probability: {probability*100:.2f}%")
        st.info("Clinical Conclusion: Patient metrics align with heart disease patterns. (मरीज में हृदय रोग के लक्षण दिख रहे हैं।)")
    else:
        st.success(f"Low Risk Detected. Probability: {(1-probability)*100:.2f}% (Negative)")
        st.info("Clinical Conclusion: Parameters are within normal clinical ranges. (हृदय के मानक सामान्य हैं।)")