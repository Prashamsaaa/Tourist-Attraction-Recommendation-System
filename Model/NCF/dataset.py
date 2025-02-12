import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from config import *

class NCFDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def load_and_preprocess_data(merged_path):
    print("Loading Data>>>")
    merged_df = pd.read_csv(merged_path)

    # Rename for consistency
    merged_df.rename(columns={"ID": "id"}, inplace=True)

    # Drop any rows with missing values in critical columns
    merged_df = merged_df.dropna(subset=["user_id", "id", "rating"])

    # Convert data types
    merged_df["user_id"] = merged_df["user_id"].astype(int)
    merged_df["id"] = merged_df["id"].astype(int)
    merged_df["rating"] = merged_df["rating"].astype(float)

    # Remove duplicate ratings
    merged_df = merged_df.drop_duplicates(subset=["user_id", "id"])

    # Encode user and place IDs
    user_encoder = LabelEncoder()
    place_encoder = LabelEncoder()
    merged_df["user_id"] = user_encoder.fit_transform(merged_df["user_id"])
    merged_df["id"] = place_encoder.fit_transform(merged_df["id"])

    return merged_df, user_encoder, place_encoder




def create_train_test_split(ratings_df, test_size=TEST_SIZE, seed=SEED):
    """
    Splits the ratings DataFrame into training and testing sets, ensuring each user has at least one training sample.

    Args:
        ratings_df (pd.DataFrame): The input DataFrame containing at least 'user_id' and ratings columns.
        test_size (float): Proportion of data to be used for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
    """
    print(f"Splitting data: {(1 - test_size) * 100:.1f}% train, {test_size * 100:.1f}% test")

    train_list = []
    test_list = []

    for user_id, user_data in ratings_df.groupby("user_id"):
        if len(user_data) < 2:
            train_list.append(user_data)  # Keep users with only 1 rating in train set
        else:
            train, test = train_test_split(user_data, test_size=test_size, random_state=seed)
            train_list.append(train)
            test_list.append(test)

    train_df = pd.concat(train_list, ignore_index=True)

    # Handle case where all users have <2 ratings and test_list is empty
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=ratings_df.columns)

    return train_df, test_df



def create_dataset_and_loaders(train_df, test_df, batch_size):
    print("Creating Dataset and Loading the Data...")

    train_dataset = NCFDataset(
        torch.tensor(train_df["user_id"].values, dtype=torch.long),
        torch.tensor(train_df["id"].values, dtype=torch.long),
        torch.tensor(train_df["rating"].values, dtype=torch.float32),
    )

    test_dataset = NCFDataset(
        torch.tensor(test_df["user_id"].values, dtype=torch.long),
        torch.tensor(test_df["id"].values, dtype=torch.long),
        torch.tensor(test_df["rating"].values, dtype=torch.float32),
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
