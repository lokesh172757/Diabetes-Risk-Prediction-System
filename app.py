import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# 2. Load the Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('diabetes_model_rf.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# 3. App Title
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Adjust the sliders below to simulate a patient record.")
st.divider()

# 4. Sidebar Inputs
st.sidebar.header("Patient Vitals")

def user_input_features():
    # These match your training columns exactly
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', 0, 200, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin Level (mu U/ml)', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.372)
    age = st.sidebar.slider('Age (years)', 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    return pd.DataFrame(data, index=[0])

# Get input
input_df = user_input_features()

# 5. Show Input Data
st.subheader("Current Patient Data")
st.write(input_df)

# 6. Predict
if st.button("Analyze Risk", type="primary"):
    if model is not None:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Result")
        if prediction[0] == 1:
            st.error(f"⚠️ High Risk of Diabetes Detected ({probability:.2%})")
        else:
            st.success(f"✅ Low Risk / Healthy ({probability:.2%})")
    else:
        st.error("Error: Could not load 'best_diabetes_model.pkl'. Is it in the same folder?")