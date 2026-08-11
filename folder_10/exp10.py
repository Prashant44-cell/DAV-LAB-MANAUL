"""
Ex. No: 10
DATA VISUALIZATION – NORMAL CURVES ON UCI DIABETES DATASET

AIM:
To visualize the distribution of key numerical attributes in the UCI Diabetes dataset using normal curves.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")

print("Normal Curve - Glucose Mean:", uci_diabetes["Glucose"].mean())
print("Normal Curve - BMI Mean:", uci_diabetes["BMI"].mean())

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(uci_diabetes["Glucose"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["Glucose"].min(), uci_diabetes["Glucose"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["Glucose"].mean(), uci_diabetes["Glucose"].std()), 'r')
plt.title("Normal Curve - Glucose")

plt.subplot(1, 2, 2)
sns.histplot(uci_diabetes["BMI"], kde=True, stat="density", linewidth=0)
x = np.linspace(uci_diabetes["BMI"].min(), uci_diabetes["BMI"].max(), 100)
plt.plot(x, norm.pdf(x, uci_diabetes["BMI"].mean(), uci_diabetes["BMI"].std()), 'r')
plt.title("Normal Curve - BMI")

plt.tight_layout()
plt.savefig("normal_curves.png")
plt.close()
print("Normal curves generated and saved.")