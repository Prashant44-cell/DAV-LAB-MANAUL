"""
Ex. No: 7
BIVARIATE ANALYSIS: LINEAR AND LOGISTIC REGRESSION MODELING

AIM:
To perform Bivariate Analysis on the UCI Diabetes Dataset and Pima Indians Diabetes Dataset using Linear Regression and Logistic Regression.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

def linear_regression_analysis(df, x_column, y_column, dataset_name):
    X = df[[x_column]]
    Y = df[y_column]
    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)
    r2 = r2_score(Y, Y_pred)
    print(f"Linear Regression ({dataset_name} - Predicting {y_column} using {x_column}): R² Score: {r2:.4f}")

linear_regression_analysis(uci_diabetes, "Glucose", "BMI", "UCI")
linear_regression_analysis(pima_diabetes, "Glucose", "BMI", "Pima")

def logistic_regression_analysis(df, features, target, dataset_name):
    X = df[features]
    Y = df[target]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Logistic Regression ({dataset_name} - Predicting {target}): Accuracy Score: {accuracy:.4f}")

features = ["Glucose", "BloodPressure", "BMI", "Age"]
target = "Outcome"
logistic_regression_analysis(uci_diabetes, features, target, "UCI")
logistic_regression_analysis(pima_diabetes, features, target, "Pima")