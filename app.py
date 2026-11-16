
import streamlit as st
import pandas as pd
import joblib

# Load saved models
heating_model = joblib.load("heating_model.pkl")
cooling_model = joblib.load("cooling_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Energy Consumption Prediction")
st.write("Enter the building details to predict Heating and Cooling Load.")

st.subheader("Building Details")

X1 = st.number_input("Relative Compactness", min_value=0.0, value=0.90)
X2 = st.number_input("Surface Area", min_value=0.0, value=563.5)
X3 = st.number_input("Wall Area", min_value=0.0, value=318.5)
X4 = st.number_input("Roof Area", min_value=0.0, value=122.5)
X5 = st.number_input("Overall Height", min_value=0.0, value=7.0)
X6 = st.selectbox("Orientation", [2, 3, 4, 5])
X7 = st.number_input("Glazing Area", min_value=0.0, value=0.10)
X8 = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

if st.button("Predict Energy Load"):
    input_data = pd.DataFrame(
        [[X1, X2, X3, X4, X5, X6, X7, X8]],
        columns=["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    )

    scaled_data = scaler.transform(input_data)

    heating_prediction = heating_model.predict(scaled_data)
    cooling_prediction = cooling_model.predict(scaled_data)

    st.success("Prediction Completed!")

    st.write("Predicted Heating Load:", round(float(heating_prediction[0]), 2))
    st.write("Predicted Cooling Load:", round(float(cooling_prediction[0]), 2))
