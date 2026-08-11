"""
Ex. No: 16
TIME SERIES ANALYSIS AND ARIMA FORECASTING

AIM:
To perform Time Series Analysis on diabetes-related datasets, identifying trends, seasonality, and patterns in glucose levels over time.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

diabetes_data = pd.read_csv("diabetes9.csv")
print("Time Series Data Head:\n", diabetes_data.head())

train_size = int(len(diabetes_data) * 0.8)
train, test = diabetes_data['Glucose'][:train_size], diabetes_data['Glucose'][train_size:]

model = ARIMA(train, order=(5, 1, 0))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test))

plt.figure(figsize=(12, 5))
plt.plot(range(len(test)), test, label="Actual", color="blue")
plt.plot(range(len(test)), forecast, label="Forecast", color="red")
plt.title("ARIMA Model Forecasting")
plt.legend()
plt.savefig("arima_forecast.png")
plt.close()

print("ARIMA Forecasting completed and plot saved.")