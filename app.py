import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

def load_data(city):
    model=joblib.load(f"models/{city}_model.pkl")
    scaler=joblib.load(f"models/{city}_scaler.pkl")
    return model,scaler


city=st.selectbox("Select City", options=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"])
model, scaler=load_data(city)


st.title("🏠 House Price Prediction App")

st.info(
    """
    This model is trained on **synthetic but realistic residential house price data
    from suburban Mumbai, India**.

    - Reflects typical Mumbai house pricing trends
    - Intended for **educational demonstration only**
"""
)

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
               **Estimated Price in Mumbai: {price_text}**""")
    
    sizes=np.linspace(500,5000,20)
    input_data=np.array([[s,rooms,age] for s in sizes])
    input_scaled=scaler.transform(input_data)
    prices=model.predict(input_scaled)

    fig ,ax=plt.subplots()
    ax.plot(sizes,prices,marker='o')
    ax.set_xlabel("House size(sqft)")
    ax.set_ylabel("Price")
    ax.set_title(f"Price vs Size in {city}")
    st.pyplot(fig)
