import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

X=np.array([
    [800,1,20],
    [1000,1,15],
    [1200,2,10],
    [1500,2,7],
    [1800,3,5],
    [2000,3,2]
])

y=np.array([40,50,65,80,95,110])

model=LinearRegression()
model.fit(X,y)

st.title("House Price Prediction App")
size=st.number_input("House Size (sqft)", min_value=500, max_value=5000, value=1000)
floors=st.number_input("Number of floors", min_value=1, max_value=5, value=1)
age=st.number_input("Age of House(years)",min_value=0, max_value=50,value=5)

if st.button("Predict Prize"):
    input_data=np.array([[size,floors,age]])
    prediction=model.predict(input_data)
    st.success(f"Estimated Price: Rs {prediction[0]:.2f} Lakhs")