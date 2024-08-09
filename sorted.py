import pandas as pd

def sort_csv(file_path):
  df = pd.read_csv('Problem Statement 2_ Data set.xlsx - Data Set.csv')

  # Sort the DataFrame by Machine, Component, and Parameter
  df = df.sort_values(by=['Machine', 'Component', 'Parameter'])

  return df

# Example usage:
file_path = "your_data.csv"
sorted_df = sort_csv(file_path)

# Save the sorted DataFrame to a new CSV file (optional)
sorted_df.to_csv("sorted_data.csv", index=False)