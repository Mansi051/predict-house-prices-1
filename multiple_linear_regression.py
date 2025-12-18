import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
os.makedirs("models",exist_ok=True)

x_base=np.array([
    [800, 2, 20],
    [1000, 2, 15],
    [1200, 3, 10],
    [1500, 3, 7],
    [1800, 4, 5],
    [2200, 4, 2]
])
city_prices = {
    "Mumbai":     [180, 240, 310, 390, 480, 580],
    "Delhi":      [150, 210, 270, 340, 420, 510],
    "Bangalore":  [130, 180, 235, 300, 370, 450],
    "Chennai":    [110, 155, 205, 260, 320, 390],
    "Hyderabad":  [105, 150, 200, 255, 315, 380],
    "Pune":       [100, 145, 190, 245, 300, 360]
}

for city,prices in city_prices.items():
    y=np.array(prices)
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x_base)


    model=LinearRegression()
    model.fit(x_scaled,y)

    joblib.dump(model,f"models/{city}_model.pkl")
    joblib.dump(scaler, f"models/{city}_scaler.pkl")
    print(f" {city} model trained & saved")
print("\n All city models trained succesfully")