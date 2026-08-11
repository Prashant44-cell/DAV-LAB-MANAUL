"""
Ex. No: 11
HYPOTHESIS TESTING – Z-TEST ON UCI DIABETES DATASET

AIM:
To perform a Z-test on the UCI Diabetes dataset to determine whether the mean Glucose level significantly differs from a given population mean (e.g., 100).
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from statsmodels.stats.weightstats import ztest

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
z_stat, p_value = ztest(uci_diabetes["Glucose"], value=100)

print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The mean Glucose level is significantly different from 100.")
else:
    print("Fail to reject the null hypothesis: No significant difference in mean Glucose level.")