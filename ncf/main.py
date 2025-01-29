from data_statistics import check_data_statistics
from train_model import train_model
from recommendation import generate_recommendations
from sklearn.model_selection import train_test_split
from config import batch_size


# Import additional necessary libraries
import torch
from torch.utils.data import DataLoader
from NeuMF import NCF  # Replace with actual model
from dataset import NCFDataset  # Replace with actual dataset

# Split into train and test by *users*
user_ids = ratings_df['user_id'].unique()
train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=30)

train_df = ratings_df[ratings_df['user_id'].isin(train_users)].copy()
test_df = ratings_df[ratings_df['user_id'].isin(test_users)].copy()



#  Convert split DataFrames to tensors
train_user_ids = torch.tensor(train_df['user_id'].values, dtype=torch.long)
train_item_ids = torch.tensor(train_df['id'].values, dtype=torch.long)
train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)

test_user_ids = torch.tensor(test_df['user_id'].values, dtype=torch.long)
test_item_ids = torch.tensor(test_df['id'].values, dtype=torch.long)
test_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float32)


# Create dataset and dataloaders
train_dataset = NCFDataset(train_user_ids, train_item_ids, train_ratings)[:2]
test_dataset = NCFDataset(test_user_ids, test_item_ids, test_ratings)[:2]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Use the defined batch_size
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Use the defined batch_size





# Model, optimizer, and loss function setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCF().to(device)  # add parameters here
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss()

# Dataset statistics
train_df, test_df = train_dataset.get_dataframe(), test_dataset.get_dataframe()
check_data_statistics(train_df, test_df)

# Training the model
train_model(model, train_loader, test_loader, 10, criterion, optimizer, device, top_k=10, num_items=len(train_dataset))

torch.save(model.state_dict(), "trained_model.pth")

# Recommendation example
user_id = 1
recommendations = generate_recommendations(
    model, user_id, len(train_dataset), top_k=5, device=device, place_encoder=train_dataset.place_encoder, 
    attraction_df=train_dataset.attraction_df, user_item_matrix=train_dataset.user_item_matrix
)
print(recommendations)
