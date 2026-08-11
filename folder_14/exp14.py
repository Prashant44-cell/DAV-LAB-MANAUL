"""
Ex. No: 14
BUILDING AND VALIDATING LINEAR MODELS

AIM:
To build and validate Linear Regression Models using the UCI and Pima Indians Diabetes datasets.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Age"

X_uci = uci_diabetes[features]
y_uci = uci_diabetes[target]
X_pima = pima_diabetes[features]
y_pima = pima_diabetes[target]

X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(X_uci, y_uci, test_size=0.2, random_state=42)
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_pima, y_pima, test_size=0.2, random_state=42)

m_uci = LinearRegression().fit(X_tr_u, y_tr_u)
m_pima = LinearRegression().fit(X_tr_p, y_tr_p)

p_uci = m_uci.predict(X_te_u)
p_pima = m_pima.predict(X_te_p)

print("UCI Diabetes Dataset - Linear Regression Results:")
print(f"R² Score: {r2_score(y_te_u, p_uci):.4f}, MSE: {mean_squared_error(y_te_u, p_uci):.4f}, MAE: {mean_absolute_error(y_te_u, p_uci):.4f}")

print("\nPima Indians Diabetes Dataset - Linear Regression Results:")
print(f"R² Score: {r2_score(y_te_p, p_pima):.4f}, MSE: {mean_squared_error(y_te_p, p_pima):.4f}, MAE: {mean_absolute_error(y_te_p, p_pima):.4f}")