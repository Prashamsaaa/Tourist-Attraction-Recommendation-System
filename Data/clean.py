import pandas as pd

# # Load the CSV files
# file1 = pd.read_csv('Data/FinalDataset/actual_description.csv')  # First CSV with id, name, description
# file2 = pd.read_csv('Data/FinalDataset/places.csv')  # Second CSV with id, name, address, latitude, longitude, location types

# file1['id'] = file1['id'].astype(str)
# file2['id'] = file2['id'].astype(str)
# # Merge the two DataFrames on 'id'
# merged_df = pd.merge(file1[['id', 'name', 'description']], file2[['id', 'address', 'latitude', 'longitude', 'location','types']], on='id', how='left')

# merged_df.to_csv('merged_file.csv', index=False)

# Load the merged CSV file
merged_df = pd.read_csv('merged_file.csv')

# Clean any ** characters from the entire DataFrame
merged_df = merged_df.replace(r'\*\*', '', regex=True)

# Save the cleaned data to a new CSV file
merged_df.to_csv('merged_cleaned_file.csv', index=False)