import pandas as pd
from sklearn.linear_model import LinearRegression

file = pd.read_csv("weatherHistory.csv")
weather_features = ['Wind Speed (km/h)','Humidity']
X = file[weather_features]
y = file.Temperature

pred_model = LinearRegression()
pred_model.fit(X, y)
# here we put wind speed and humidity e.g., 15 , 0.13
y_predict = pred_model.predict([[15, 0.13]])
print(y_predict)
