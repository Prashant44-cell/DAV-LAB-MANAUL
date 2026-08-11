"""
Ex. No: 5
EXPLORING DESCRIPTIVE ANALYTICS USING THE IRIS DATASET

AIM:
To explore descriptive analytics using the Iris dataset with Python's Pandas and Seaborn libraries.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('iris_dataset(2d).csv')
print("Basic Information:")
df.info()
print("\nSummary Statistics:")
print(df.describe())
print("\nSpecies Count:")
print(df['species'].value_counts())

df.hist(figsize=(8, 6), edgecolor='black')
plt.suptitle('Feature Distributions')
plt.savefig('iris_hist.png')
plt.close()

sns.boxplot(data=df, x='species', y='sepal length (cm)')
plt.title('Sepal Length Comparison')
plt.savefig('iris_box.png')
plt.close()

sns.pairplot(df, hue='species')
plt.savefig('iris_pair.png')
plt.close()
print("Plots created successfully.")