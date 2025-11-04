import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Solar Power Prediction", page_icon="🔆", layout="centered")

st.title("Solar Power Generation Predictor")
st.write("Enter the weather parameters below and get predicted **power-generated**.")

@st.cache_resource
def load_artifacts():
    model_path = "best_model.pkl"
    scaler_path = "scaler.pkl"
    if not os.path.exists(model_path):
        st.error("best_model.pkl not found. Please upload the trained model file to this folder or your repo.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error("scaler.pkl not found. Please upload the fitted scaler file to this folder or your repo.")
        st.stop()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

FEATURES = [
    "distance-to-solar-noon",
    "temperature",
    "wind-direction",
    "wind-speed",
    "sky-cover",
    "visibility",
    "humidity",
    "average-wind-speed-(period)",
    "average-pressure-(period)"
]

st.sidebar.header("Input Parameters")
def sidebar_inputs():
    vals = {}
    vals["distance-to-solar-noon"] = st.sidebar.number_input("distance-to-solar-noon (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
    vals["temperature"] = st.sidebar.number_input("temperature (°F)", min_value=-50, max_value=150, value=70, step=1)
    vals["wind-direction"] = st.sidebar.number_input("wind-direction (deg)", min_value=0, max_value=360, value=90, step=1)
    vals["wind-speed"] = st.sidebar.number_input("wind-speed (mph)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    vals["sky-cover"] = st.sidebar.number_input("sky-cover (0–100)", min_value=0, max_value=100, value=20, step=1)
    vals["visibility"] = st.sidebar.number_input("visibility (miles)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    vals["humidity"] = st.sidebar.number_input("humidity (%)", min_value=0, max_value=100, value=50, step=1)
    vals["average-wind-speed-(period)"] = st.sidebar.number_input("average-wind-speed-(period)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    vals["average-pressure-(period)"] = st.sidebar.number_input("average-pressure-(period)", min_value=0.0, max_value=40.0, value=29.8, step=0.01)
    return vals

user_vals = sidebar_inputs()

def predict_power(input_dict):
    row = pd.DataFrame([[input_dict[f] for f in FEATURES]], columns=FEATURES)
    X_scaled = scaler.transform(row)
    pred = model.predict(X_scaled)[0]
    return float(pred)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔮 Predict"):
        pred = predict_power(user_vals)
        st.success(f"Estimated Power Generated: **{pred:,.0f}** units")

with col2:
    if st.button("✨ Reset to Defaults"):
        st.experimental_rerun()

with st.expander("🔧 See input as table"):
    st.dataframe(pd.DataFrame([user_vals]))

st.markdown("""
---
**Notes**
- Make sure `best_model.pkl` and `scaler.pkl` are in the same folder.
- Feature order must match training:  
`distance-to-solar-noon, temperature, wind-direction, wind-speed, sky-cover, visibility, humidity, average-wind-speed-(period), average-pressure-(period)`.
""")
