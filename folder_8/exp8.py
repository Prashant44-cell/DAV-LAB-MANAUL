"""
Ex. No: 8
STATISTICAL ANALYSIS USING DIABETES DATASETS – MULTIPLE REGRESSION ANALYSIS

AIM:
To perform multiple regression analysis on the UCI Diabetes and Pima Indians Diabetes datasets to predict BMI based on multiple independent variables.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

features = ["Glucose", "BloodPressure", "Age"]
target = "BMI"

def multiple_regression_analysis(df, dataset_name):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{dataset_name} - Multiple Regression R² Score: {r2:.4f}")

multiple_regression_analysis(uci_diabetes, "UCI Diabetes Dataset")
multiple_regression_analysis(pima_diabetes, "Pima Indians Diabetes Dataset")