# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:15:06 2024

@author: Shiyun Liu
"""
import numpy as np
import yfinance as yf
import pandas as pd
import openpyxl  # Create the Ticker object
aapl = yf.Ticker("AAPL")
# Get historical market data
# hist = aapl.history(start="2022-10-01", end="2024-04-02")
hist = aapl.history(start="1980-12-01", end="2024-07-02")
# Remove timezone information
hist.index = hist.index.tz_localize(None)
hist.to_excel("AAPL_stockprices.xlsx")
# Save the data to an Excel file
hist.iloc[:-1].to_excel("AAPL_input.xlsx")

# %% binary classification
# Load the Excel file
file_path = 'AAPL_stockprices.xlsx'
df = pd.read_excel(file_path)

# Extract the 'Close' column
close_column = df['Close']

# Calculate differences and update the column
close_column_diff = close_column.diff()
# Assign differences starting from the second row
close_column.iloc[1:] = close_column_diff.iloc[1:]

# Update the original DataFrame with the modified 'Close' column
df['Close'] = close_column
# Modify 'Close' column based on conditions
df['Close'] = (df['Close'] > 0).astype(int)
# Save the updated DataFrame to Excel
updated_file_path = 'AAPL_output.xlsx'
df['Close'].iloc[1:].to_excel(updated_file_path, index=False)
