"""
Ex. No: 13
PERFORM ANOVA ON DIABETES DATASETS

AIM:
To perform ANOVA (Analysis of Variance) on the UCI Diabetes and Pima Indians Diabetes datasets to analyze differences between multiple group means.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from scipy.stats import f_oneway

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

numerical_columns = ["Glucose", "BloodPressure", "BMI"]
anova_results = {}
for col in numerical_columns:
    f_stat, p_value = f_oneway(uci_diabetes[col], pima_diabetes[col])
    anova_results[col] = {"F-statistic": f_stat, "P-value": p_value}

anova_df = pd.DataFrame(anova_results).T
print("ANOVA Results:\n", anova_df)