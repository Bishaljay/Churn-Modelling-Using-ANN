import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ===== Load label encoders =====
if os.path.exists("geo_encoder.pkl") and os.path.exists("gender_encoder.pkl"):
    geo_encoder = joblib.load("geo_encoder.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    print("✅ Label encoders loaded.")
else:
    # Create new encoders with default categories (demo mode)
    print("⚠ Encoders not found. Creating demo label encoders.")
    geo_encoder = LabelEncoder()
    geo_encoder.fit(["France", "Germany", "Spain"])
    gender_encoder = LabelEncoder()
    gender_encoder.fit(["Male", "Female"])
    joblib.dump(geo_encoder, "geo_encoder.pkl")
    joblib.dump(gender_encoder, "gender_encoder.pkl")
    print("✅ New encoders created and saved.")

# ===== Load model =====
if os.path.exists("ann_model.h5"):
    model = load_model("ann_model.h5")
    print("✅ Model loaded.")
else:
    st.error("❌ ann_model.h5 not found. Please provide a trained model file.")
    st.stop()

# ===== Streamlit UI =====
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction using ANN")

st.sidebar.header("Enter Customer Information")

def user_input():
    CreditScore = st.sidebar.slider('Credit Score', 350, 850, 600)
    Geography = st.sidebar.selectbox('Geography', ['France', 'Germany', 'Spain'])
    Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    Age = st.sidebar.slider('Age', 18, 92, 35)
    Tenure = st.sidebar.slider('Tenure (Years with Bank)', 0, 10, 5)
    Balance = st.sidebar.number_input('Account Balance', 0.0, 250000.0, 100000.0)
    NumOfProducts = st.sidebar.selectbox('Number of Products', [1, 2, 3, 4])
    HasCrCard = st.sidebar.selectbox('Has Credit Card?', [0, 1])
    IsActiveMember = st.sidebar.selectbox('Is Active Member?', [0, 1])
    EstimatedSalary = st.sidebar.number_input('Estimated Salary', 10000.0, 200000.0, 50000.0)

    data = {
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    }

    return pd.DataFrame(data)

input_df = user_input()

if st.button("Predict Churn"):
    # ===== Apply label encoding to categorical columns =====
    input_df['Geography'] = geo_encoder.transform(input_df['Geography'])
    input_df['Gender'] = gender_encoder.transform(input_df['Gender'])

    # ===== Ensure feature order matches training =====
    feature_order = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    processed = input_df[feature_order].values  # shape will be (1, 10)

    # ===== Make prediction =====
    prediction = model.predict(processed)
    churn = int(prediction[0][0] > 0.5)

    st.subheader("🔍 Prediction Result:")
    if churn == 1:
        st.error("❌ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
    st.markdown(f"**Churn Probability:** `{prediction[0][0]:.2f}`")
