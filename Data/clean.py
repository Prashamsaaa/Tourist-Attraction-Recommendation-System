import pandas as pd
import numpy as np
from difflib import SequenceMatcher

def clean_attractions_data(input_file, clean_output_file, uncleaned_output_file):
    """
    Clean the attractions dataset, remove redundant entries, and save both cleaned and uncleaned data.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    def is_valid_description(desc):
        if pd.isna(desc) or not isinstance(desc, str):
            return False
        if desc.strip() in ['Kathmandu', 'Nepal', 'Kathmandu, Nepal']:
            return False
        if len(desc.strip().split()) < 10:
            return False
        return True
    
    # Create mask for valid descriptions
    valid_mask = df['description'].apply(is_valid_description)
    
    # Split into cleaned and uncleaned dataframes
    cleaned_df = df[valid_mask].copy()
    uncleaned_df = df[~valid_mask].copy()
    
    # Clean up formatting
    cleaned_df['description'] = cleaned_df['description'].apply(lambda x: x.strip())
    cleaned_df['name'] = cleaned_df['name'].apply(lambda x: str(x).strip())
    
    def text_similarity(text1, text2):
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()
    
    def find_redundant_entries(df):
        """Find redundant entries based on name, description, and location similarity."""
        redundant_indices = set()
        keep_indices = set()
        n = len(df)
        
        # Create arrays for faster comparison
        names = df['name'].values
        descs = df['description'].values
        lats = df['latitude'].values
        longs = df['longitude'].values
        ratings = df['rating'].fillna(0).values
        
        for i in range(n):
            if i in redundant_indices:
                continue
                
            for j in range(i + 1, n):
                if j in redundant_indices:
                    continue
                    
                # Check if locations are very close (within ~100 meters)
                if (not pd.isna(lats[i]) and not pd.isna(lats[j]) and 
                    not pd.isna(longs[i]) and not pd.isna(longs[j])):
                    dist = np.sqrt((lats[i] - lats[j])**2 + (longs[i] - longs[j])**2)
                    location_similar = dist < 0.001  # roughly 100 meters
                else:
                    location_similar = False
                
                # Check name and description similarity
                name_similarity = text_similarity(names[i], names[j])
                desc_similarity = text_similarity(descs[i], descs[j])
                
                # If entries are similar enough, keep the one with higher rating
                if (location_similar and (name_similarity > 0.8 or desc_similarity > 0.8)):
                    if ratings[i] >= ratings[j]:
                        redundant_indices.add(j)
                        keep_indices.add(i)
                    else:
                        redundant_indices.add(i)
                        keep_indices.add(j)
                        break
        
        return redundant_indices

    # Find and remove redundant entries
    print("Finding redundant entries...")
    redundant_indices = find_redundant_entries(cleaned_df)
    
    # Move redundant entries to uncleaned_df
    redundant_mask = cleaned_df.index.isin(redundant_indices)
    redundant_df = cleaned_df[redundant_mask].copy()
    redundant_df['removal_reason'] = 'Redundant entry'
    cleaned_df = cleaned_df[~redundant_mask]
    
    # Add removal reasons to uncleaned data
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
    
    # Combine uncleaned and redundant entries
    uncleaned_df = pd.concat([uncleaned_df, redundant_df])
    
    # Sort cleaned data by rating
    cleaned_df['rating'] = cleaned_df['rating'].fillna(0)
    cleaned_df = cleaned_df.sort_values('rating', ascending=False)
    
    # Save both datasets
    cleaned_df.to_csv(clean_output_file, index=False)
    uncleaned_df.to_csv(uncleaned_output_file, index=False)
    
    # Print summary statistics
    print("\nData Cleaning Summary:")
    print("-" * 50)
    print(f"Original number of entries: {len(df)}")
    print(f"Cleaned entries: {len(cleaned_df)}")
    print(f"Removed entries: {len(uncleaned_df)}")
    print(f"Redundant entries removed: {len(redundant_indices)}")
    print("\nBreakdown of removed entries:")
    print(uncleaned_df['removal_reason'].value_counts())
    
    return cleaned_df, uncleaned_df

# Usage
if __name__ == "__main__":
    input_file = "Data/FinalDataset/attractions_uncleaned.csv"
    clean_output_file = "attractions_cleaned.csv"
    uncleaned_output_file = "attractions_uncleaned.csv"
    
    cleaned_data, uncleaned_data = clean_attractions_data(
        input_file, 
        clean_output_file, 
        uncleaned_output_file
    )