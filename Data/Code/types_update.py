import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = 'Data/FinalDataset/Data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Check for null values in the DataFrame
null_cells = df.isnull().sum()

# Display the result
print("Number of null values in each column:")
print(null_cells)
