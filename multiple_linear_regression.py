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
    "Mumbai":     [90, 115, 150, 190, 235, 280],
    "Delhi":      [80, 105, 140, 175, 215, 255],
    "Bangalore":  [70, 95, 125, 160, 195, 230],
    "Chennai":    [60, 85, 115, 145, 175, 205],
    "Hyderabad":  [55, 80, 110, 140, 170, 200],
    "Pune":       [50, 75, 100, 130, 160, 190]
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