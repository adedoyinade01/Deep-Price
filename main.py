import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

# App Config
st.set_page_config(page_title="House Price Explorer", layout="wide")
st.title("🏡 House Price Data Explorer")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/kc_house_data.csv")

df = load_data()

# Load Model and Preprocessing Objects
model = load_model("model/house_price_model.keras")
scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")  # same columns used during training

# 1. Dataset Preview
st.subheader("📊 Dataset Preview")
if st.checkbox("Show raw data"):
    # Remove less relevant columns for cleaner display
    display_df = df.drop(columns=['lat', 'long', 'sqft_living15', 'sqft_lot15'])
    st.dataframe(display_df.head(10))

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

# 4. Predict New House Price
st.subheader("🎯 Predict on a New House")

with st.form("prediction_form"):
    sqft_living = st.number_input("Living Space (sqft)", min_value=500, max_value=10000, value=2000)
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=50000, value=5000)
    bedrooms = st.slider("Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Bathrooms", 1, 5, 2)
    floors = st.slider("Floors", 1, 3, 1)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)
    grade = st.slider("Grade", 1, 13, 7)
    sqft_above = st.number_input("Sqft Above", 500, 10000, 1500)
    sqft_basement = st.number_input("Sqft Basement", 0, 5000, 500)
    yr_renovated = st.number_input("Year Renovated", 0, 2022, 0)

    month_names = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Month Built", month_names, index=5)
    month = month_names.index(selected_month) + 1
    yr_built = st.number_input("Year Built", 1900, 2022, 1990)

    submitted = st.form_submit_button("Predict")\
        
       
    # Hidden fields with reasonable defaults (model still needs these)
    lat = float(df['lat'].mean())  # Default to Seattle area average
    long = float(df['long'].mean())  # Default to Seattle area average
    sqft_living15 = sqft_living  # Default to same as main living space
    sqft_lot15 = sqft_lot  # Default to same as main lot size
    # year = st.number_input("Year Built", 2014, 2024, 2022)
    year = datetime.now().year
    
    if submitted:
        
        input_values = [
            bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
            grade, sqft_above, sqft_basement, yr_built, yr_renovated,
            lat, long, sqft_living15, sqft_lot15, month, year
        ]
        
        input_data = pd.DataFrame([input_values], columns=feature_columns)

        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0][0]
        st.success(f"Predicted House Price: ${int(predicted_price):,}")

# 5. Optional: Model Evaluation (y_test vs predictions) — Placeholder
# You can update this with your actual predictions if you saved y_test and predictions
st.subheader("📉 Model Evaluation Example (Placeholder)")
st.markdown("_You can replace this with your real test data and predictions_")

y_test = np.random.randint(100000, 1000000, 100)
predictions = y_test + np.random.normal(0, 100000, 100)

fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.scatter(y_test, predictions, alpha=0.6)
ax4.plot(y_test, y_test, 'r', label='Perfect Prediction')
ax4.set_xlabel("Actual Prices")
ax4.set_ylabel("Predicted Prices")
ax4.legend()
st.pyplot(fig4)
