from config import Config
from preprocess import load_and_preprocess_data, split_data
from dataset import create_dataloaders
from NeuMF import NCF
import torch.optim as optim
import torch.nn as nn
from train_model import train_epoch, validate
from recommendation import Recommender

def main():
    # print(Config.DATA_PATH)
    
    # Load and preprocess data
    df, user_mapping, item_mapping = load_and_preprocess_data(Config.DATA_PATH)
    train_df, val_df, test_df = split_data(df, Config.TRAIN_RATIO, Config.VAL_RATIO)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, Config.BATCH_SIZE
    )
    hidden_layers = [2**(7 - i) for i in range(Config.NUM_LAYERS)]
    
    # Initialize model
    model = NCF(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        latent_dim=Config.LATENT_DIM,
        hidden_layers= hidden_layers,
    )
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Create recommender
    recommender = Recommender(model, item_mapping)
    
    return recommender

if __name__ == "__main__":
    main()
