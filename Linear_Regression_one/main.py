import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file = pd.read_csv("weatherHistory.csv")
x = file['Wind Speed (km/h)'].values.reshape(-1, 1)
y = file['Temperature (C)']

lr_model = LinearRegression()
lr_model.fit(x, y)
# here put the wind speed e.g., 5
y_pred = lr_model.predict(np.array([5]).reshape(1, 1))
print(y_pred)
