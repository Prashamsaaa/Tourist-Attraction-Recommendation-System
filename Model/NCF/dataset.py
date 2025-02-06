import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


class NCFDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


def load_and_preprocess_data(ratings_path, attraction_path):
    print("Loading Data>>>")
    attraction_df = pd.read_csv(attraction_path)
    ratings_df = pd.read_csv(ratings_path)

    ratings_df["id"] = ratings_df["id"].fillna(-1)
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "id"])

    user_encoder = LabelEncoder()
    place_encoder = LabelEncoder()
    ratings_df["user_id"] = user_encoder.fit_transform(ratings_df["user_id"])
    ratings_df["id"] = place_encoder.fit_transform(ratings_df["id"])

    return ratings_df, attraction_df, user_encoder, place_encoder


def create_train_test_split(ratings_df, test_size):
    print(
        f"Splitting data into train size of {(1- test_size)*100} and test size of {test_size * 100}"
    )
    user_ids = ratings_df["user_id"].unique()
    train_users, test_users = train_test_split(
        user_ids, test_size=test_size, random_state=30
    )
    return (
        ratings_df[ratings_df["user_id"].isin(train_users)].copy(),
        ratings_df[ratings_df["user_id"].isin(test_users)].copy(),
    )


def create_dataset_and_loaders(train_df, test_df, batch_size):
    print(f"Creating Dataset and Loading the Data.......")
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
