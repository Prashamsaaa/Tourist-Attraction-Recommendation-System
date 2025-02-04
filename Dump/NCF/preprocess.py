import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    
    # Create sequential user and item mappings
    unique_users = df['user_id'].unique()
    unique_items = df['id'].unique()
    
    user_mapping = {id_: idx for idx, id_ in enumerate(sorted(unique_users))}
    item_mapping = {id_: idx for idx, id_ in enumerate(sorted(unique_items))}
    
    # Map IDs to indices
    df['user_id'] = df['user_id'].map(user_mapping)
    df['id'] = df['id'].map(item_mapping)
    
    return df, user_mapping, item_mapping

def split_data(df, train_ratio=0.8, val_ratio=0.1):
    train_df, temp_df = train_test_split(df, train_size=train_ratio)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio/(1-train_ratio))
    return train_df, val_df, test_df
