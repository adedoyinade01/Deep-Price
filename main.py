import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Basic setup
st.set_page_config(page_title="House Price Explorer", layout="wide")
st.title("🏡 House Price Data Explorer")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/kc_house_data.csv")

df = load_data()

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

# 4. Model Evaluation (sample scatter for y_test vs predictions)
st.subheader("📉 Model Evaluation Example")

# Dummy values (replace with your actual y_test and predictions)
y_test = np.random.randint(100000, 1000000, 100)
predictions = y_test + np.random.normal(0, 100000, 100)

fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.scatter(y_test, predictions, alpha=0.6)
ax4.plot(y_test, y_test, 'r', label='Perfect Prediction')
ax4.set_xlabel("Actual Prices")
ax4.set_ylabel("Predicted Prices")
ax4.legend()
st.pyplot(fig4)

# 5. Predict New House Price
st.subheader("🧠 Predict on a New House")

# Sample form (extend as needed)
with st.form("prediction_form"):
    sqft_living = st.number_input("Living Space (sqft)", min_value=500, max_value=10000, value=2000)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
    floors = st.slider("Number of Floors", 1, 3, 1)
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Dummy prediction logic — replace with your model
        dummy_price = sqft_living * 200 + bedrooms * 10000 + bathrooms * 5000 + floors * 7000
        st.success(f"Predicted Price: ${int(dummy_price):,}")
