import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title of the web app
st.title('Customer Churn Prediction')

# Load the trained model, scaler, and column names
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# Define the function to make predictions
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Define the input fields
st.header('Input Features')
tenure = st.number_input('Tenure', min_value=0, max_value=100, value=0)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=1000.0, value=0.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=0.0)

# Collect the input features into a DataFrame
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Reindex the input data to match the training data's columns
input_data = input_data.reindex(columns=columns, fill_value=0)

# Standardize the input data using the saved scaler
input_data = scaler.transform(input_data)

# Display the input data
st.write('Input Data:')
st.write(input_data)

# Make predictions
if st.button('Predict'):
    prediction = predict_churn(input_data)
    st.write(f'The prediction is: {"Churn" if prediction[0] else "No Churn"}')
