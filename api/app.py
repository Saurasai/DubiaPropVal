import os
import sys
import time
import requests
import pandas as pd
import streamlit as st
from streamlit_extras.let_it_rain import rain

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.valuation import classify_price, validate_price
from src.predict import predict_price

# ----------------- Model Download -------------------
MODEL_URL = "https://www.dropbox.com/scl/fi/xr2z9v6e7ru31fs4u5m2v/price_predictor.pkl?rlkey=0r4df2i2iprbuseg22bkloliu&st=4r38e071&dl=1"
MODEL_PATH = "models/price_predictor.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Dropbox..."):
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        st.success("✅ Model downloaded successfully!")

download_model()

# ------------------ Load Data -----------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_science_challenge_data.csv')
    return pd.read_csv(csv_path)

df = load_data()
neighbourhoods = sorted(df['neighbourhood'].dropna().unique())
buildings = sorted(df['building'].dropna().unique(), key=lambda x: int(''.join(filter(str.isdigit, str(x))) or 0))

# ------------------ UI Setup ------------------------
# Animated Title
title_placeholder = st.empty()
colors = ["#FF5733", "#33FF57", "#3357FF", "#F333FF"]
emoji = "🏙️"
for i in range(4):
    color = colors[i % len(colors)]
    title_placeholder.markdown(
        f"<h1 style='color:{color}; text-align:center'>{emoji} Dubai Property Price Prediction & Valuation Tool</h1>",
        unsafe_allow_html=True
    )
    time.sleep(0.25)

# Sidebar
with st.sidebar:
    st.header("📌 Tips & Info")
    st.markdown("""
    - Select the **neighbourhood** and **building**.
    - Size is in square feet (e.g., 1200).
    - Bedrooms/Bathrooms can be decimals (e.g., 1.5).
    - Listing price is used for valuation logic.
    """)

# ------------------ Form ------------------------
with st.form(key='price_form'):
    st.subheader("Enter Property Details")
    neighbourhood = st.selectbox("🏘️ Neighbourhood", neighbourhoods)
    building = st.selectbox("🏢 Building", buildings)
    size = st.number_input("📐 Size (sqft)", min_value=0.0, step=0.1)
    bedrooms = st.number_input("🛏️ Bedrooms", min_value=0.0, step=1.0)
    bathrooms = st.number_input("🛁 Bathrooms", min_value=0.0, step=1.0)
    listing_price = st.number_input("💰 Listing Price (AED)", min_value=0.0, step=1000.0)

    submit_button = st.form_submit_button(label='🔍 Predict Price')

# ------------------ Prediction Logic ------------------------
if submit_button:
    features = {
        'neighbourhood': neighbourhood,
        'building': building,
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms
    }

    with st.spinner("Predicting property price..."):
        predicted_price = predict_price(features)
        valuation = classify_price(predicted_price, listing_price)
        validation_msg = validate_price(features, predicted_price)

    # 🎉 Confetti celebration
    rain(emoji="🎉", font_size=54, falling_speed=5, animation_length=2)
    st.balloons()

    st.success(f"### 🧮 Predicted Price: AED {predicted_price:,.2f}")
    st.info(f"### 📊 Valuation: {valuation}")

    if validation_msg:
        st.warning("### ⚠️ Validation Messages:")
        # Fix: ensure validation_msg is iterable (not a string)
        if isinstance(validation_msg, str):
            validation_msg = [validation_msg]
        for msg in validation_msg:
            st.write(f"• {msg}")
