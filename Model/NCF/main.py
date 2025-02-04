import logging
import pandas as pd
from config import MODEL_CONFIG, TRAINING_CONFIG
from dataset import load_dataset, preprocess_data, NCFDataset
from preprocess import encode_features
from gmf_mlp import GMF, MLP
from NeuMF import NeuMF
from train_model import train_model
from recommendation import generate_recommendations
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Step 1: Load and preprocess the dataset
    data_path = "../Data/PreparedData.csv"
    ratings_path = "../Data/all_ratings.csv"
    data, ratings_df = load_dataset(data_path, ratings_path)

    if ratings_df is None:
        logging.error("No ratings data available")
        return

    data = preprocess_data(data, ratings_df)
    data, user_encoder, item_encoder = encode_features(data)

    # Load the dataset into a DataFrame for mapping item IDs to names
    places_df = pd.read_csv(data_path)

    # Step 2: Initialize models
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    gmf_model = GMF(num_users, num_items, MODEL_CONFIG['embedding_dim']).to(device)
    mlp_model = MLP(
        num_users=num_users,
        num_items=num_items,
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_layers=MODEL_CONFIG['layers']
    ).to(device)
    
    neummf_model = NeuMF(gmf_model, mlp_model).to(device)

    # Create dataloaders
    train_dataset = NCFDataset(
        user_ids=torch.tensor(data['user_id'].values),
        item_ids=torch.tensor(data['item_id'].values),
        ratings=torch.tensor(data['rating'].values, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True
    )

    # Initialize loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        neummf_model.parameters(),
        lr=TRAINING_CONFIG['learning_rate']
    )

    # Train the model
    train_model(
        model=neummf_model,
        train_loader=train_loader,
        test_loader=None,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=TRAINING_CONFIG['epochs'],
        device=device
    )

    # Generate recommendations for a specific user
    user_id = 34  # Example user ID
    recommendations = generate_recommendations(neummf_model, user_id, num_items=num_items, device=device)

    # Map recommended IDs to their names using the dataset
    recommended_places = places_df[places_df['ID'].isin(recommendations)][['ID', 'Name']]

    # Convert to list of tuples (ID, Name)
    recommendations_with_names = list(zip(recommended_places['ID'], recommended_places['Name']))

    # Log recommendations with both IDs and names
    logging.info(f"Recommendations for User {user_id}: {recommendations_with_names}")

if __name__ == "__main__":
    main()