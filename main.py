import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# Basic setup
st.set_page_config(page_title="House Price Explorer", layout="wide")
st.title("🏡 House Price Data Explorer")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/kc_house_data.csv")

df = load_data()

# Load model and scaler
model = load_model('model/house_price_model.keras')
scaler = joblib.load('model/scaler.pkl')

# 1. Dataset Preview
st.subheader("📊 Dataset Preview")
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))

# 2. Visual Analysis
st.subheader("📈 Data Visualization")

tab1, tab2 = st.tabs(["Price vs Living Space", "Bedrooms vs Price"])

with tab1:
    st.markdown("#### Scatterplot: Price vs Sqft Living")
    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.scatterplot(x='price', y='sqft_living', data=df, ax=ax1)
    st.pyplot(fig1)

with tab2:
    st.markdown("#### Boxplot: Bedrooms vs Price")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.boxplot(x='bedrooms', y='price', data=df, ax=ax2)
    st.pyplot(fig2)

# 3. Geographical Properties
st.subheader("🌍 Geographical Price Distribution")
fig3, ax3 = plt.subplots(figsize=(10,6))
sns.scatterplot(x='price', y='long', data=df, ax=ax3)
st.pyplot(fig3)

# 4. Model Evaluation (real predictions)
st.subheader("📉 Model Evaluation Example")

# Create X and y to get real predictions
X = df.drop('price', axis=1)
y = df['price']

# Scale X
X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)

# Show sample
sample_y = y[:100]
sample_pred = predictions[:100]

fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.scatter(sample_y, sample_pred, alpha=0.6)
ax4.plot(sample_y, sample_y, 'r', label='Perfect Prediction')
ax4.set_xlabel("Actual Prices")
ax4.set_ylabel("Predicted Prices")
ax4.legend()
st.pyplot(fig4)

# 5. Predict New House Price
st.subheader("🧐 Predict on a New House")

with st.form("prediction_form"):
    input_data = {}
    for col in df.drop('price', axis=1).columns:
        default_val = float(df[col].mean())
        if df[col].dtype in [int, float]:
            input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), default_val)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        predicted_price = model.predict(scaled_input)
        st.success(f"Predicted Price: ${predicted_price[0][0]:,.2f}")
