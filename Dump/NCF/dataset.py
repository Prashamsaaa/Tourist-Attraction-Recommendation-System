import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TourismDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = ratings_df['user_id'].values
        self.items = ratings_df['id'].values
        self.ratings = ratings_df['rating'].values
        
    def __len__(self):
        return len(self.ratings)
        
    def __getitem__(self, idx):
        return {
            'user': torch.tensor(self.users[idx], dtype=torch.long),
            'item': torch.tensor(self.items[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float)
        }

def create_dataloaders(train_df, val_df, test_df, batch_size):
    train_dataset = TourismDataset(train_df)
    val_dataset = TourismDataset(val_df)
    test_dataset = TourismDataset(test_df)
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size)
    )
