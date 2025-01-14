import pandas as pd
import re

def clean_attractions_data(input_file, clean_output_file, uncleaned_output_file):
    """
    Clean the attractions dataset and save both cleaned and uncleaned data separately.
    
    Parameters:
    input_file (str): Path to the input CSV file
    clean_output_file (str): Path to save the cleaned CSV file
    uncleaned_output_file (str): Path to save the uncleaned/removed entries
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Function to check if a description is valid
    def is_valid_description(desc):
        if pd.isna(desc) or not isinstance(desc, str):
            return False
        
        # Remove common location markers that aren't real descriptions
        if desc.strip() in ['Kathmandu', 'Nepal', 'Kathmandu, Nepal']:
            return False
        
        # Check if description has meaningful length (at least 10 words)
        words = desc.strip().split()
        if len(words) < 10:
            return False
            
        return True
    
    # Create mask for valid descriptions
    valid_mask = df['description'].apply(is_valid_description)
    
    # Split into cleaned and uncleaned dataframes
    cleaned_df = df[valid_mask].copy()
    uncleaned_df = df[~valid_mask].copy()
    
    # Clean up description formatting for cleaned data
    cleaned_df['description'] = cleaned_df['description'].apply(lambda x: x.strip())
    
    # Remove duplicate entries based on name and description
    cleaned_df = cleaned_df.drop_duplicates(subset=['name', 'description'])
    
    # Sort by rating (descending) and fill any missing ratings with 0
    cleaned_df['rating'] = cleaned_df['rating'].fillna(0)
    cleaned_df = cleaned_df.sort_values('rating', ascending=False)
    
    # Add reason for removal to uncleaned data
    def get_removal_reason(row):
        desc = row['description']
        if pd.isna(desc):
            return "Missing description"
        if not isinstance(desc, str):
            return "Invalid description type"
        if desc.strip() in ['Kathmandu', 'Nepal', 'Kathmandu, Nepal']:
            return "Generic location description"
        if len(desc.strip().split()) < 10:
            return "Description too short (< 10 words)"
        return "Other"
    
    uncleaned_df['removal_reason'] = uncleaned_df.apply(get_removal_reason, axis=1)
    
    # Save both datasets
    cleaned_df.to_csv(clean_output_file, index=False)
    uncleaned_df.to_csv(uncleaned_output_file, index=False)
    
    # Print summary statistics
    print("\nData Cleaning Summary:")
    print("-" * 50)
    print(f"Original number of entries: {len(df)}")
    print(f"Cleaned entries: {len(cleaned_df)}")
    print(f"Removed entries: {len(uncleaned_df)}")
    print("\nBreakdown of removed entries:")
    print(uncleaned_df['removal_reason'].value_counts())
    
    return cleaned_df, uncleaned_df

# Usage
if __name__ == "__main__":
    input_file = "Data/FinalDataset/attractions_final.csv"
    clean_output_file = "attractions_cleaned.csv"
    uncleaned_output_file = "attractions_uncleaned.csv"
    
    cleaned_data, uncleaned_data = clean_attractions_data(
        input_file, 
        clean_output_file, 
        uncleaned_output_file
    )