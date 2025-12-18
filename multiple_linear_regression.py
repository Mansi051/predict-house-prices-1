import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
os.makedirs("models",exist_ok=True)

x_base=np.array([
    [800, 1, 20],
    [1000, 1, 15],
    [1200, 2, 10],
    [1500, 2, 5],
    [1800, 3, 5],
    [2000, 3, 2]
])
city_prices={
    "Mumbai":     [160, 200, 220, 280, 340, 420],
    "Delhi":      [120, 180, 200, 250, 310, 380],
    "Bangalore":  [90, 100, 160, 190, 240, 290],
    "Kolkata":    [80,87,95,110,150,180],
    "Hyderabad":  [80, 98, 110, 120, 190, 220],
    "Pune":       [90, 96, 105,140,210,250]
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