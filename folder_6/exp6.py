"""
Ex. No: 6
STATISTICAL ANALYSIS USING DIABETES DATASETS - UNIVARIATE ANALYSIS

AIM:
To analyze the Diabetes dataset from UCI and the Pima Indians Diabetes dataset using univariate statistical methods.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

print("UCI Diabetes Dataset Sample:\n", uci_diabetes.head())
print("\nPima Indians Diabetes Dataset Sample:\n", pima_diabetes.head())

numerical_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def univariate_analysis(df, columns):
    stats = {}
    for col in columns:
        stats[col] = {
            "Mean": np.mean(df[col]),
            "Median": np.median(df[col]),
            "Mode": df[col].mode()[0],
            "Variance": np.var(df[col], ddof=1),
            "Standard Deviation": np.std(df[col], ddof=1),
            "Skewness": skew(df[col]),
            "Kurtosis": kurtosis(df[col])
        }
    return pd.DataFrame(stats).T

uci_stats = univariate_analysis(uci_diabetes, numerical_columns)
pima_stats = univariate_analysis(pima_diabetes, numerical_columns)

print("\nUCI Diabetes Dataset Statistics:\n", uci_stats)
print("\nPima Indians Diabetes Dataset Statistics:\n", pima_stats)