import streamlit as st
import numpy as np
import pickle

# Load model
with open("New_RF_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if used
try:
    with open("New_scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

st.title("Malnutrition Prediction")
st.write("Enter child details to predict the malnutrition status.")

# User inputs
Age = st.number_input("Age", min_value=6.0, max_value=59.0, value=7.0)
weight= st.number_input("Weight (kg)", min_value=6.1, max_value=20.0, value=7.0)
#bloodpressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
Height= st.number_input("Height (cm)", min_value=60.3, max_value=120.0, value=70.0)

gender = st.selectbox("Gender", ["Female", "Male"])
Gender = 1 if gender == "Male" else 0

#
# northeast → all zeros

# Combine inputs


#
# Combine inputs correctly
input_data = np.array([[Age, Gender, weight, Height]])

# Apply scaling if used
if scaler:
    input_data = scaler.transform(input_data)

# Predict

if st.button("Predict Malnutrition Status"):
    # Predict using model
    prediction = model.predict(input_data)

    # Convert prediction to integer (0,1,2)
    pred_class = int(round(prediction[0]))  # round in case it's slightly float

    # Map numeric output to user-friendly text
    status_map = {0: "Normal", 1: "Moderate", 2: "Severe"}
    result = status_map[pred_class]

    # Show result with color coding
    if result == "Normal":
        st.success(f"Estimated Malnutrition Status: {result}")
    elif result == "Moderate":
        st.warning(f"Estimated Malnutrition Status: {result}")
    else:
        st.error(f"Estimated Malnutrition Status: {result}")
    
    # Optional: Show prediction probability %
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        prob_map = {status_map[i]: round(prob[i]*100, 2) for i in range(len(prob))}
        st.info(f"Prediction Confidence: {prob_map}")

