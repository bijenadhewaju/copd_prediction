# Import the libraries
import pandas as pd
import pickle
import streamlit as st
import numpy as np
import os

#Load the trained model
model_file_path = os.path.join(os.path.dirname(__file__), 'Best_Random_Forest_Model.pkl')
with open(model_file_path, 'rb') as f:
    model = pickle.load(f)

#Streamlit App
def main():
    st.title("COPD Prediction Dashboard")

    #User Input
    st.sidebar.header("User Input")

    age = st.sidebar.slider('Age', 30, 80, 50)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    bmi = st.sidebar.slider('BMI', 10, 40, 25)
    smoking_status = st.sidebar.selectbox('Smoking Status', ['Current', 'Former', 'Never'])
    biomass_fuel_exposure = st.sidebar.selectbox('Biomass Fuel Exposure', ['Yes', 'No'])
    occupational_exposure = st.sidebar.selectbox('Occupational Fuel Exposure', ['Yes', 'No'])
    family_copd_history = st.sidebar.selectbox('Family COPD History', ['Yes', 'No'])
    air_pollution_level = st.sidebar.slider('Air Pollution Level', 0, 300, 50)
    respiratory_infection = st.sidebar.selectbox('Respiratory Infections in Childhood', ['Yes', 'No'])
    location = st.sidebar.selectbox('Location', ['Kathmandu', 'Lalitpur', 'Pokhara', 'Butwal', 'Nepalgunj', 'Hetauda', 'Chitwan', 'Dharan', 'Biratnagar'])

    # Process the input data
    input_data = {
        'Age': [age],
        'Biomass_Fuel_Exposure': [biomass_fuel_exposure],
        'Occupational_Exposure': [occupational_exposure],
        'Family_History_COPD': [family_copd_history],
        'BMI': [bmi],
        'Air_Pollution_Level': [air_pollution_level],
        'Respiratory_Infections_Childhood': [respiratory_infection],
        'Pollution_Risk_Score': [0],  #dummy
        'Smoking_Status_Encoded': [smoking_status],
        'Gender_Encoded': [gender],
        'Smoking_Pollution_Interaction': [0],  #dummy
        'Location': [location]
       }

    # Convert the data to a dataframe
    input_df = pd.DataFrame(input_data)

    # Encoding the data
    input_df['Gender_Encoded'] = input_df['Gender_Encoded'].map({'Male': 1, 'Female': 0})
    input_df['Smoking_Status_Encoded'] = input_df['Smoking_Status_Encoded'].map({'Current': 1, 'Former': 0.5, 'Never': 0})
    input_df['Biomass_Fuel_Exposure'] = input_df['Biomass_Fuel_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Occupational_Exposure'] = input_df['Occupational_Exposure'].map({'Yes': 1, 'No': 0})
    input_df['Family_History_COPD'] = input_df['Family_History_COPD'].map({'Yes': 1, 'No': 0})
    input_df['Respiratory_Infections_Childhood'] = input_df['Respiratory_Infections_Childhood'].map({'Yes': 1, 'No': 0})

    # Calculate Pollution Risk Score based on Air Pollution Level
    input_df['Pollution_Risk_Score'] = np.where(input_df['Air_Pollution_Level'] > 150, 1, 0)

    # Calculate Smoking Pollution Interaction
    input_df['Smoking_Pollution_Interaction'] = input_df['Smoking_Status_Encoded'] * input_df['Air_Pollution_Level']

    #Drop the Location column
    input_df_encoded = pd.get_dummies(input_df, columns=['Location'], drop_first=False)

    # List of columns used during model training
    # New columns based on values
    columns_during_training = ['Location_Biratnagar', 'Location_Butwal', 'Location_Chitwan', 
                           'Location_Dharan', 'Location_Hetauda', 'Location_Kathmandu', 
                           'Location_Lalitpur', 'Location_Nepalgunj', 'Location_Pokhara']
    
    
    # Add missing columns that the model was trained on
    for col in columns_during_training:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0



    # Ensure the correct column order as seen during training
    input_df_encoded = input_df_encoded[columns_during_training]

    input_df = pd.concat([input_df, input_df_encoded], axis=1)
    input_df = input_df.drop(columns=['Location'])
    # Prediction
    if st.button('Predict COPD'):
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.write('Prediction: COPD Detected')
        else:
            st.write('Prediction: No COPD Detected')

if __name__ == '__main__':
    main()