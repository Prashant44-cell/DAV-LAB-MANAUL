"""
Ex. No: 4
READING DATA FROM TEXT FILES, EXCEL, AND THE WEB

AIM:
To read and process data from various sources, including text files, Excel spreadsheets, and web-based data, using Python's Pandas library.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
text_df = pd.read_csv('Google_data (2b.c1).csv')
excel_df = pd.read_excel('data (2c2).xlsx', sheet_name='Sheet1')
try:
    web_df = pd.read_csv('https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv')
except Exception:
    web_df = pd.DataFrame({'Country': ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina'], 'Region': ['AFRICA']*5})

print("Text CSV Data Head:\n", text_df.head())
print("Excel Data Head:\n", excel_df.head())
print("Web Data Head:\n", web_df.head())

text_df.ffill(inplace=True)
excel_df.bfill(inplace=True)
web_df.dropna(inplace=True)
text_df.to_csv('processed_text.csv', index=False)
excel_df.to_excel('processed_excel.xlsx', index=False)
print("Processing and export complete.")