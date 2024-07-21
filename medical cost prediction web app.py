# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:53:28 2024

@author: mukes
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('D:/ML/trained_model.sav','rb'))

# Define a function to convert categorical inputs to numeric
def preprocess_input(age, sex, bmi, children, smoker, region):
    # Convert age, bmi, children to numeric
    age = int(age)
    bmi = float(bmi)
    children = int(children)
    
    # Convert sex to numeric
    if sex.lower() == 'male':
        sex = 0
    else:
        sex = 1
    
    # Convert smoker to numeric
    if smoker.lower() == 'yes':
        smoker = 0
    else:
        smoker = 1
    
    # Convert region to numeric (using one-hot encoding for example)
    regions = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
    region = regions.get(region.lower(), -1)
    if region == -1:
        raise ValueError("Invalid region value")
    
    return [age, sex, bmi, children, smoker, region]


#creating a function for prediction
def cost_prediction(input_data):
    

    # changing input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    return ('The insurance cost is USD ', prediction[0])



def main():
    
    #giving a title
    st.title('Medical Insurance Cost Prediction')
    
    #getting the input data from the user
    #age,sex,bmi,children,smoker,region
    
    age = st.text_input('Age of the person')
    sex = st.text_input('Male or Female')
    bmi = st.text_input('bmi value')
    children = st.text_input('No. of Children')
    smoker = st.text_input('Smoker - yes or no')
    region = st.text_input('Region - southeast, southwest, northeast, northwest')
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Cost Prediction'):
        #diagnosis = cost_prediction([age,sex,bmi,children,smoker,region])
        try:
            # Preprocess the input
            input_data = preprocess_input(age, sex, bmi, children, smoker, region)
            # Make prediction
            diagnosis = cost_prediction(input_data)
        except ValueError as e:
            diagnosis = f"Error in input values: {e}"
        
    st.success(diagnosis)  
    
    
    
if __name__ == '__main__':
    main()    
        
    
