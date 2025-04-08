import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv('data/kc_house_data.csv')

# Load model and scaler
model = load_model('model/house_price_model.h5')
scaler = joblib.load('model/scaler.pkl')

st.title("🏡 Seattle House Price Prediction")

st.sidebar.header("Enter Features of the House")

def user_input():
    features = {}
    for col in df.drop('price', axis=1).columns:
        features[col] = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    return pd.DataFrame([features])

user_data = user_input()

# Section 1: Data Overview
st.subheader("Data Overview")
st.write(df.head())

# Section 2: Visualizations
st.subheader("Scatterplot: Price vs Living Area")
fig1, ax1 = plt.subplots()
sns.scatterplot(x='price', y='sqft_living', data=df, ax=ax1)
st.pyplot(fig1)

st.subheader("Boxplot: Bedrooms vs Price")
fig2, ax2 = plt.subplots()
sns.boxplot(x='bedrooms', y='price', data=df, ax=ax2)
st.pyplot(fig2)

st.subheader("Scatterplot: Price vs Longitude")
fig3, ax3 = plt.subplots()
sns.scatterplot(x='price', y='long', data=df, ax=ax3)
st.pyplot(fig3)

# Section 3: Predict Price
st.subheader("Predicted House Price")

if st.button("Predict"):
    scaled_input = scaler.transform(user_data)
    prediction = model.predict(scaled_input)
    st.success(f"Estimated Price: ${prediction[0][0]:,.2f}")
