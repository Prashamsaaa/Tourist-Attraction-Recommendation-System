import os
import torch
import numpy as np
import random
import json
from tqdm import tqdm
from config import *
from models import NCF
from dataset import (
    load_and_preprocess_data,
    create_train_test_split,
    create_dataset_and_loaders,
)
from train_eval import train_model, evaluate_topn
from utils import plot_loss_curves, calculate_average_metrics
from dynamic_model_manager import DynamicModelManager
from recommendation import generate_recommendations
from utils import set_seed
import matplotlib.pyplot as plt

def setup_seed(seed):
    set_seed(seed)  # Your existing set_seed function
    # Additional seed settings for data loaders
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def generate_new_folder(base_path="Model/outputs"):
    """Generate a new folder name like 'test1', 'test2', etc., and create it."""
    folder_num = 1
    while os.path.exists(f"{base_path}/test{folder_num}"):
        folder_num += 1
    new_folder = f"{base_path}/test{folder_num}"
    os.makedirs(new_folder)
    return new_folder


def save_config(config, folder_path):
    """Save the configuration settings to a JSON file in the specified folder."""
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved at {config_path}")


def main():
    # Set seed at the very beginning
    setup_seed(SEED)

    # Load and preprocess data
    ratings_df, attraction_df, user_encoder, place_encoder = load_and_preprocess_data(
        "Model/Data/all_ratings.csv", "Model/Data/PreparedData.csv"
    )
    num_users = len(user_encoder.classes_)
    num_items = len(place_encoder.classes_)

    # Split data with fixed seed
    train_df, test_df = create_train_test_split(ratings_df, test_size=TEST_SIZE)
    train_loader, test_loader = create_dataset_and_loaders(
        train_df, test_df, BATCH_SIZE
    )

    print("Initializing the NCF model...........")

    # Initialize model
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)
    print(f"Model: {model}")

    # Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    train_losses, test_losses, metrics = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        DEVICE,
        num_items,
    )

    # Generate new folder to save outputs
    output_folder = generate_new_folder(base_path="Model/outputs")
    print(f"Saving outputs to {output_folder}")

    # Save model to the new folder
    model_save_path = os.path.join(output_folder, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # Save config to the new folder
    save_config({
        "NUM_EPOCHS": NUM_EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "LATENT_DIM": LATENT_DIM,
        "TOP_K": TOP_K,
        "NUM_LAYERS": NUM_LAYERS,
        "TEST_SIZE": TEST_SIZE,
        "SEED": SEED,
        "HIDDEN_LAYERS": HIDDEN_LAYERS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT_RATE": DROPOUT_RATE,
        "DEVICE": str(DEVICE),
        "MODEL_SAVE_PATH": model_save_path,
        "last_epoch_metrics": metrics  # Saving last epoch metrics
    }, output_folder)

    print("-------------------Training Completed-----------------------")
    print("Printing Average Metrics and Displaying Loss Curves")
    
    # Calculate and print average metrics
    calculate_average_metrics(
        metrics["hit_rates"],
        metrics["ndcgs"],
        metrics["rmses"],
        metrics["precisions"],
        metrics["recalls"],
        metrics["maes"],
        TOP_K,
    )

    # Save the loss plot
    plot_loss_curves(
        train_losses=train_losses, test_losses=test_losses, num_epochs=NUM_EPOCHS
    )

    # Initialize Model Manager
    model_manager = DynamicModelManager(
        model, optimizer, criterion, user_encoder, place_encoder, backup_frequency= 2
    )

    # Generate recommendations
    original_user_id = 5
    encoded_user_id = user_encoder.transform([original_user_id])[0]
    recommendations = generate_recommendations(
        model_manager.model,
        encoded_user_id,
        num_items,
        TOP_K,
        DEVICE,
        place_encoder,
        attraction_df,
    )

    print(f"\nTop {TOP_K} recommendations for user {original_user_id}:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['Name']} (Confidence: {rec['confidence']:.3f})")


if __name__ == "__main__":
    main()
