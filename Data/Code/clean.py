import pandas as pd

# # Load the CSV files
# file1 = pd.read_csv('Data/FinalDataset/ai_description.csv')  # First CSV with id, name, description
# file2 = pd.read_csv('Data/FinalDataset/places.csv')  # Second CSV with id, name, address, latitude, longitude, location types

# file1['id'] = file1['id'].astype(str)
# file2['id'] = file2['id'].astype(str)
# # Merge the two DataFrames on 'id'
# merged_df = pd.merge(file1[['id', 'name', 'description']], file2[['id', 'address', 'latitude', 'longitude', 'location','types']], on='id', how='left')

# merged_df.to_csv('merged_file.csv', index=False)

# Load the merged CSV file
# merged_df = pd.read_csv('merged_file.csv')

# # Clean any ** characters from the entire DataFrame
# merged_df = merged_df.replace(r'\*\*', '', regex=True)

# # Save the cleaned data to a new CSV file
# merged_df.to_csv('merged_cleaned_file.csv', index=False)



# # Function to determine 'types' based on 'description'
# def assign_type(description):
#     description = str(description).lower()  # Convert to lowercase to handle case insensitivity
    
#     if 'temple' in description:
#         return 'Temple'
#     elif 'park' in description:
#         return 'Park'
#     elif 'waterfall' in description:
#         return 'Waterfall'
#     elif 'monument' in description:
#         return 'Monument'
#     elif 'historic' in description or 'site' in description:
#         return 'Historic Site'
#     else:
#         return 'Other'  # Default category if none of the keywords are found

# # Apply the function only where 'types' is empty (NaN)
# merged_df['types'] = merged_df['types'].fillna(merged_df.loc[merged_df['types'].isna(), 'description'].apply(assign_type))



# # Save the updated data to a new CSV file
# merged_df.to_csv('merged_filled_file.csv', index=False)
import pandas as pd

# Load the CSV file
file_path = 'Data/FinalDataset/merged_all.csv'  # Replace with your file's path
data = pd.read_csv(file_path)

# Check for non-integer values in the 'id' column
try:
    data['id'] = data['id'].astype(int)
except ValueError:
    print("The 'id' column contains invalid values. Fixing them...")

    # Option 1: Remove rows with invalid 'id' values
    data = data[pd.to_numeric(data['id'], errors='coerce').notna()]
    data['id'] = data['id'].astype(int)

    # Option 2: Replace invalid values with a default (e.g., 0)
    # data['id'] = pd.to_numeric(data['id'], errors='coerce').fillna(0).astype(int)

# Sort the DataFrame by the 'id' column
sorted_data = data.sort_values(by='id')

# Save the sorted CSV
sorted_data.to_csv('sorted_file.csv', index=False)

print("File has been sorted and saved as 'sorted_file.csv'.")

