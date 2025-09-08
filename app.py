import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("breast.pkl")
    scaler = joblib.load("scalar.pkl")
    return model, scaler

model, scaler = load_model()

# Define the labels and feature names
slider_labels = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

st.title("Breast Cancer Diagnosis")
st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar.")

# Sidebar sliders for input features
st.sidebar.header("Input Features")
input_data = []
for label, feature in slider_labels:
    val = st.sidebar.slider(label, 0.0, 100.0, 10.0)
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

if st.sidebar.button("Predict Diagnosis"):
    prediction = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.success(f"Prediction: Malignant (Probability: {pred_proba:.2%})")
    else:
        st.info(f"Prediction: Benign (Probability: {1-pred_proba:.2%})")

# Data visualization section
st.header("🔍 Data Visualization")
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    return df

df = load_data()

# Feature selection for visualization
feature_names = [f[1] for f in slider_labels]
selected_feature = st.selectbox("Select a feature to visualize", feature_names)

# Plot distribution for selected feature, colored by diagnosis
fig, ax = plt.subplots()
for diag, color, label in zip(['M', 'B'], ['red', 'green'], ['Malignant', 'Benign']):
    ax.hist(df[df['diagnosis'] == diag][selected_feature], bins=30, alpha=0.6, color=color, label=label)
ax.set_title(f"Distribution of {selected_feature} by Diagnosis")
ax.set_xlabel(selected_feature)
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Add a title
st.set_page_config(page_title="Breast Cancer Diagnosis",
                    page_icon="👩‍⚕️", 
                    layout="wide", 
                    initial_sidebar_state="expanded")
