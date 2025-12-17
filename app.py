import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


X=np.array([
    [800,1,20],
    [1000,1,15],
    [1200,2,10],
    [1500,2,7],
    [1800,3,5],
    [2000,3,2]
])

y=np.array([180,220,260,310,360,420])

model=LinearRegression()
model.fit(X,y)

st.title("🏠 House Price Prediction App")

st.info(
    """
    This model is trained on **synthetic but realistic residential house price data
    from suburban Mumbai, India**.

    - Prices are in **Lakhs INR**
    - Reflects typical Mumbai apartment pricing trends
    - Intended for **educational demonstration only**
"""
)

st.divider()
st.subheader("Enter House Details")


size=st.slider("House Size (sqft)", min_value=500, max_value=5000, value=1000, step=50)
floors=st.selectbox("Number of floors", options=[1,2,3,4,5])
age=st.slider("Age of House(years)",min_value=0, max_value=50,value=5)

if st.button("Predict Prize"):
    input_data=np.array([[size,floors,age]])
    prediction=model.predict(input_data)
    st.success(f"""
               **Estimated Price in Mumbai: Rs {prediction[0]:.2f} Lakhs**""")