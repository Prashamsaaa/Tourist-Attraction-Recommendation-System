def calculate_statistics(data):
    """
    Calculate and display statistics for the dataset.
    """
    print(f"Records: {len(data)}")
    print(f"Unique users: {data['user_id'].nunique()}")
    print(f"Unique items: {data['item_id'].nunique()}")

def print_data_statistics(df):
    print("Rows:", len(df))
    print("Rating distribution:\n", df['rating'].value_counts())