#app to deploy model
import numpy as np
!pip install streamlit
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
filename = 'crop_recommendation_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a function for prediction
def crop_prediction(input_data):
    # Change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction[0]

def main():
    # Giving a title
    st.title('Crop Recommendation System')

    # Getting the input data from the user
    N = st.text_input("Nitrogen")
    P = st.text_input("Phosphorus")
    K = st.text_input("Potassium")
    temperature = st.text_input("Temperature")
    humidity = st.text_input("Humidity")
    ph = st.text_input("pH")
    rainfall = st.text_input("Rainfall")

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Crop Recommendation'):
        input_data = [N, P, K, temperature, humidity, ph, rainfall]
        diagnosis = crop_prediction(input_data)

    st.success(diagnosis)

if __name__ == '__main__':
    main()