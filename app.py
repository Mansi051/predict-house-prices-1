import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

def load_data(city):
    model=joblib.load(f"models/{city}_model.pkl")
    scaler=joblib.load(f"models/{city}_scaler.pkl")
    return model,scaler

st.title("🏠 House Price Prediction App")

st.info("""
    This model estimates house prices for **premium/posh residential areas** in major Indian cities. 
    The dataset used is synthetically generated but inspired by real-world housing price trends in premium residential areas across major Indian cities. Actual prices may vary based on locality, amenities, and market conditions.
- Reflects typical housing trends across different urban areas
- Intended for **educational demonstration only**
""")


city=st.selectbox("Select City", options=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"])
model, scaler=load_data(city)

st.divider()
st.subheader("Enter House Details")


size=st.slider("House Size (sqft)", min_value=500, max_value=5000, value=1000, step=50)
rooms=st.selectbox("Number of rooms", options=[1,2,3,4,5])
age=st.slider("Age of House(years)",min_value=0, max_value=50,value=5)

if st.button("Predict Price"):
    input_data=np.array([[size,rooms,age]])
    input_scaled=scaler.transform(input_data)
    prediction=model.predict(input_scaled)[0]
    predicted_price = round(prediction, 2)

    if predicted_price>=100:
        price_text=f"₹ {predicted_price/100:.2f} Crores"
    else:
        price_text=f"₹ {predicted_price:.2f} Lakhs"
    st.success(f"""
               **Estimated Price in {city}: {price_text}**""")
    
    
