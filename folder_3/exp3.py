"""
Ex. No: 3
WORKING WITH PANDAS DATAFRAMES

AIM:
To explore and perform various DataFrame operations using Pandas, including loading datasets, data inspection, handling missing values, transformations, filtering, grouping, sorting, and saving results.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
df = pd.read_csv('data.csv')
print("First 5 rows:\n", df.head())
print("Last 5 rows:\n", df.tail())
df.info()
print("Summary statistics:\n", df.describe())
df['Rating'] = df['Rating'].fillna(df['Rating'].mean())
df['new_column'] = df['Rating'] * 2
series = df['Rating']
print("Series addition:\n", (series + 10).head())
filtered_df = df[(df['Rating'] > 4.0) & (df['Reviews'] > 500)]
print("Filtered DataFrame shape:", filtered_df.shape)
grouped = df.groupby('Category')['Rating'].mean()
print("Grouped mean:\n", grouped.head())
df_sorted = df.sort_values(by='Rating', ascending=False)
print("Sorted DataFrame:\n", df_sorted.head())
masked_df = df[df['Rating'] > df['Rating'].median()]
print("Masked DataFrame shape:", masked_df.shape)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
subset_df = df[['App', 'Category']]
subset_df.to_csv('filtered_data.csv', index=False)
print("Total sum:", df['Rating'].sum())
print("Mean:", df['Rating'].mean())
print("Standard Deviation:", df['Rating'].std())