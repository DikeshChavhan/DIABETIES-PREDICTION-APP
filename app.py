import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("data.pkl", "rb") as file:
    model = pickle.load(file)

# App title
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This application predicts whether a person is likely to have **diabetes**  
based on medical attributes.  
Please enter the details below 👇
""")

# Create input fields
pregnancies = st.number_input("👶 Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("🩸 Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("💓 Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("💪 Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("💉 Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("👨‍👩‍👧 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("🎂 Age", min_value=0, max_value=120, value=30)

# Prepare the input data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, dpf, age]])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person is **likely to have diabetes.**")
    else:
        st.success("✅ The model predicts that the person is **not likely to have diabetes.**")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Machine Learning")
