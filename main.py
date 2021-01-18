# packages import
import numpy as np
import pandas as pd

"""
Function definitions
"""
def calculate_df(dev_fact, dataframe):
  j = 2 # column counter
  while j <= len(dataframe):
    # initialization
    sum1 = 0
    sum2 = 0
    for i in range(0, len(dataframe)): # row counter
      if np.isfinite(dataframe.at[i, f'{j}']):
        sum1 = sum1 + dataframe.at[i, f'{j-1}']
        sum2 = sum2 + dataframe.at[i, f'{j}']
    dev_fact.append(sum2/sum1) # we take only the 6 digits after the ,
    j = j+1

def fill_na(column1, column2, dev_factor):
  for i in range(0, len(column2)):
    if np.isnan(column2[i]):
      column2[i] = column1[i] * dev_factor
# define the initial data
"""
This is where we define the initial dataframe to use on our prediction process with the method "Chain ladder"
"""
initial_data = pd.read_csv("data.csv")
print("-"*15, "Initial Data", "-"*15)
print(initial_data)
# build the cumulative data
print("-"*15, "Cumulative Data", "-"*15)
for i, item in initial_data.iterrows():
    initial_data.at[i]  = pd.Series(item).cumsum()
print(initial_data)
# get the development factors
dev_factors = []
calculate_df(dev_factors, initial_data)
print("-"*15, "Development Factors", "-"*15)
print(dev_factors)
# build the lower triangle of the matrix
i = 2
while i <= len(initial_data):
  fill_na(initial_data[f'{i-1}'], initial_data[f'{i}'], dev_factors[i-2])
  i = i + 1
print("-"*15, "Fitted Cumulative", "-"*15)
print(initial_data)
# Fitted incremental

# Pearson residuals