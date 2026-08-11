"""
Ex. No: 15
BUILDING AND VALIDATING LOGISTIC MODELS

AIM:
To build and validate Logistic Regression Models for predicting diabetes presence using the UCI and Pima Indians Diabetes datasets.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

uci_diabetes = pd.read_csv("uci_diabetes (3).csv")
pima_diabetes = pd.read_csv("pima_diabetes (3).csv")

features = ["Glucose", "BloodPressure", "BMI"]
target = "Outcome"

X_tr_u, X_te_u, y_tr_u, y_te_u = train_test_split(uci_diabetes[features], uci_diabetes[target], test_size=0.2, random_state=42)
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(pima_diabetes[features], pima_diabetes[target], test_size=0.2, random_state=42)

m_uci = LogisticRegression(max_iter=1000).fit(X_tr_u, y_tr_u)
m_pima = LogisticRegression(max_iter=1000).fit(X_tr_p, y_tr_p)

p_uci = m_uci.predict(X_te_u)
p_pima = m_pima.predict(X_te_p)

print("UCI Diabetes Dataset - Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_te_u, p_uci):.4f}, Precision: {precision_score(y_te_u, p_uci, zero_division=0):.4f}, Recall: {recall_score(y_te_u, p_uci, zero_division=0):.4f}, F1 Score: {f1_score(y_te_u, p_uci, zero_division=0):.4f}")

print("\nPima Indians Diabetes Dataset - Logistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_te_p, p_pima):.4f}, Precision: {precision_score(y_te_p, p_pima, zero_division=0):.4f}, Recall: {recall_score(y_te_p, p_pima, zero_division=0):.4f}, F1 Score: {f1_score(y_te_p, p_pima, zero_division=0):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_te_u, p_uci), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("UCI Diabetes - Confusion Matrix")
sns.heatmap(confusion_matrix(y_te_p, p_pima), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title("Pima Indians Diabetes - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.close()
print("Confusion matrix heatmap saved.")