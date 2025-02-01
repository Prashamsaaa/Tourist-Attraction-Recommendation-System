import pandas as pd

def check_data_statistics(train_df, test_df):
    print("\nTrain set statistics:")
    print(train_df['rating'].mean())
    print(train_df['rating'].std())
    print(train_df['rating'].value_counts())

    print("\nTest set statistics:")
    print(test_df['rating'].mean())
    print(test_df['rating'].std())
    print(test_df['rating'].value_counts())

    print("\nUnique users in train:", len(train_df['user_id'].unique()))
    print("Unique users in test:", len(test_df['user_id'].unique()))
    print("Unique items in train:", len(train_df['id'].unique()))
    print("Unique items in test:", len(test_df['id'].unique()))

    train_users_set = set(train_df['user_id'].unique())
    test_users_set = set(test_df['user_id'].unique())
    print("\nUsers only in test:", len(test_users_set - train_users_set))

    train_items_set = set(train_df['id'].unique())
    test_items_set = set(test_df['id'].unique())
    print("Items only in test:", len(test_items_set - train_items_set))
