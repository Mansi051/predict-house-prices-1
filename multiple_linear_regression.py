import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
os.makedirs("models",exist_ok=True)

x_base=np.array([
    [800, 2, 20],
    [1000, 1, 15],
    [1200, 2, 10],
    [1500, 2, 7],
    [1800, 3, 5],
    [2200, 4, 2]
])
city_prices = {
    "Mumbai":     [180, 230, 300, 380, 470, 600],
    "Delhi":      [150, 200, 260, 330, 420, 540],
    "Bangalore":  [120, 160, 220, 280, 350, 460],
    "Chennai":    [100, 140, 190, 250, 310, 400],
    "Hyderabad":  [95, 130, 180, 235, 300, 380],
    "Pune":       [90, 120, 165, 215, 270, 350]
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