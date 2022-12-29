# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:25:16 2021

@author: sachin h s
"""

import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_diabetes(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    
    """Let's predict diabetes 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Pregnancies
        in: query
        type: number
        required: true
      - name: Glucose
        in: query
        type: number
        required: true
      - name: BloodPressure
        in: query
        type: number
        required: true
      - name: SkinThickness
        in: query
        type: number
        required: true
      - name: Insulin
        in: query
        type: number
        required: true
      - name: BMI
        in: query
        type: number
        required: true
      - name: DiabetesPedigreeFunction
        in: query
        type: number
        required: true
      - name: Age
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return prediction



def main():
    st.title("diabetes prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit diabetes prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Pregnancies = st.text_input("Pregnancies","Type Here")
    Glucose = st.text_input("Glucose","Type Here")
    BloodPressure = st.text_input("BloodPressure","Type Here")
    SkinThickness = st.text_input("SkinThickness","Type Here")    
    Insulin = st.text_input("Insulin","Type Here")
    BMI = st.text_input("BMI","Type Here")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction","Type Here")
    Age = st.text_input("Age","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_diabetes(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()