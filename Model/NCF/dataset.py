import pandas as pd
import torch
from torch.utils.data import Dataset
import logging

def load_dataset(file_path, ratings_path=None):
    """Load the dataset from CSV files."""
    try:
        data = pd.read_csv(file_path)
        if ratings_path:
            ratings_df = pd.read_csv(ratings_path)
            
            # Standardize column names
            data_cols = {
                'ID': 'item_id',
                'Name': 'name',
                'Description': 'description',
                'Province': 'province',
                'Tags': 'tags'
            }
            ratings_cols = {
                'user_id': 'user_id',
                'id': 'item_id',
                'rating': 'rating'
            }
            
            data = data.rename(columns=data_cols)
            ratings_df = ratings_df.rename(columns=ratings_cols)
            
            # Ensure data types match for merging
            data['item_id'] = data['item_id'].astype(int)
            ratings_df['item_id'] = ratings_df['item_id'].astype(int)
            
            # Keep only necessary columns and merge
            merged_df = pd.merge(
                ratings_df, 
                data[['item_id', 'name', 'province', 'tags']], 
                on='item_id',
                how='inner'
            )
            
            return merged_df, ratings_df
        return data, None
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None, None

def preprocess_data(data, ratings=None):
    """Preprocess the data for training."""
    if isinstance(data, pd.DataFrame):
        if 'rating' in data.columns:
            data['interaction'] = (data['rating'] > 3).astype(int)
    return data

class NCFDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]