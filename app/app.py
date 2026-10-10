import os
import joblib
import pandas as pd
import streamlit as st

if not os.path.exists("models/best_model.pkl"):
    st.error("Model not found. Run python train.py first.")
    st.stop()

model = joblib.load("models/best_model.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("House Price Prediction")
st.write("Enter the details of the house to estimate its sale price.")

st.divider()

# Basic details
st.subheader("Basic House Details")

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=10,
    step=1
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0.5,
    max_value=8.0,
    step=0.5,
    help="0.5 means a half bathroom. Example: 2.5 = 2 full bathrooms + 1 half bathroom."
)

floors = st.number_input(
    "Floors",
    min_value=1.0,
    max_value=4.0,
    step=0.5,
    help="0.5 represents a half floor. Examples: 1.5 = one and a half floors, 2.5 = two and a half floors."
)

sqft_living = st.number_input(
    "Living Area (sq ft)",
    min_value=300,
    max_value=10000,
    step=100
)

sqft_lot = st.number_input(
    "Plot Area (sq ft)",
    min_value=500,
    max_value=1000000,
    step=500
)

waterfront = st.selectbox(
    "Waterfront Property",
    ["No", "Yes"],
    help="Select Yes if the property is located directly on a waterfront."
)

st.divider()

# Quality details
st.subheader("House Quality")

view = st.slider(
    "View Quality",
    min_value=0,
    max_value=4,
    help="0 = No view | 1 = Poor | 2 = Fair | 3 = Good | 4 = Excellent"
)

st.caption(
    "View: 0 = No view • 1 = Poor • 2 = Fair • 3 = Good • 4 = Excellent"
)

condition = st.slider(
    "House Condition",
    min_value=1,
    max_value=5,
    help="1 = Very Poor | 2 = Poor | 3 = Average | 4 = Good | 5 = Very Good"
)

st.caption(
    "Condition: 1 = Very Poor • 2 = Poor • 3 = Average • 4 = Good • 5 = Very Good"
)

grade = st.slider(
    "Construction Grade",
    min_value=1,
    max_value=13,
    help="Higher grade means better construction and design quality."
)

st.caption(
    "Grade: 1–3 Very Poor • 4–5 Below Average • 6 Average • "
    "7 Good • 8 Very Good • 9 Excellent • 10 Very High • "
    "11 Exceptional • 12 Luxury • 13 Extremely High"
)

yr_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026,
    step=1
)

st.divider()

# Prediction
if st.button("Predict House Price", use_container_width=True):

    data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": 1 if waterfront == "Yes" else 0,
        "view": view,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built
    }])

    prediction = model.predict(data)[0]

    st.success(
        f"Estimated Sale Price: **${prediction:,.0f}**"
    )