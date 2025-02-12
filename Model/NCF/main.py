import os
import torch
import numpy as np
import random
import json
import logging
from config import *
from models import NCF
from dataset import (
    load_and_preprocess_data,
    create_train_test_split,
    create_dataset_and_loaders,
)
from train_eval import train_model
from utils import (
    plot_loss_curves,
    calculate_average_metrics,
    plot_hit_rate_ndcg,
    plot_mae_curve,
)
from dynamic_model_manager import DynamicModelManager
from recommendation import generate_recommendations
from utils import set_seed

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def setup_seed(seed):
    set_seed(seed)
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
    logging.info(f"New folder created at {new_folder}")
    return new_folder

def save_config(config, folder_path):
    """Save the configuration settings to a JSON file in the specified folder."""
    config_path = os.path.join(folder_path, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Config saved at {config_path}")
    except Exception as e:
        logging.error(f"Error saving config: {e}")

def save_metrics(last_epoch_metrics, folder_path):
    metrics = {"last_epoch_metrics": last_epoch_metrics}

    metrics_path = os.path.join(folder_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved at {metrics_path}")

def save_model(model, output_folder, epoch, model_name="model.pth"):
    """Save the model checkpoint."""
    model_save_path = os.path.join(output_folder, f"{model_name}_epoch{epoch}.pth")
    try:
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved at {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def main():
    # Set seed at the very beginning
    setup_seed(SEED)

    # Load and preprocess data (from the merged CSV)
    ratings_df, user_encoder, place_encoder = load_and_preprocess_data("Model/Data/merged_output.csv")

    num_users = len(user_encoder.classes_)
    num_items = len(place_encoder.classes_)

    # Split data with fixed seed
    train_df, test_df = create_train_test_split(ratings_df, test_size=TEST_SIZE)
    train_loader, test_loader = create_dataset_and_loaders(
        train_df, test_df, BATCH_SIZE
    )
    print(f"Training data: {len(train_df)} interactions, Test data: {len(test_df)} interactions")
    logging.info("Initializing the NCF model...")

    # Initialize model
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=LATENT_DIM,
    ).to(DEVICE)
    logging.info(f"Model: {model}")

    # Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    train_losses, test_losses, metrics = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        DEVICE,
        num_items
    )

    # Generate new folder to save outputs
    output_folder = generate_new_folder(base_path="Model/outputs")

    # Save these results to config file
    save_model(model, output_folder, NUM_EPOCHS)
    save_config(
        {
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
            "MODEL_SAVE_PATH": output_folder,
        },
        output_folder,
    )
    save_metrics(
        {
            "epoch": NUM_EPOCHS,
            "hit_rate": metrics["hit_rates"][-1],
            "ndcg": metrics["ndcgs"][-1],
            "precision": metrics["precisions"][-1],
            "recall": metrics["recalls"][-1],
            "rmse": metrics["rmses"][-1],
            "mae": metrics["maes"][-1],
        },
        output_folder,
    )

    logging.info("-------------------Training Completed-----------------------")
    logging.info("Printing Average Metrics and Displaying Loss Curves")

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
        train_losses=train_losses,
        test_losses=test_losses,
        num_epochs=NUM_EPOCHS,
        output_folder=output_folder,
    )

    # Save the MAE plot
    plot_mae_curve(metrics["maes"], NUM_EPOCHS, output_folder=output_folder)

    # Save the Hit Rate and NDCG plot
    plot_hit_rate_ndcg(
        metrics["hit_rates"],
        metrics["ndcgs"],
        NUM_EPOCHS,
        TOP_K,
        output_folder=output_folder,
    )

    # Initialize Model Manager
    model_manager = DynamicModelManager(
        model, optimizer, criterion, user_encoder, place_encoder
    )

    # Generate recommendations
    original_user_id = 15  # Replace with an actual user_id
    encoded_user_id = user_encoder.transform([original_user_id])[0]
    recommendations = generate_recommendations(
        model_manager.model,
        encoded_user_id,
        ratings_df,
        TOP_K,
        DEVICE,
        place_encoder,
    )

    logging.info(f"\nTop {TOP_K} recommendations for user {original_user_id}:")
    for idx, rec in enumerate(recommendations, 1):
        logging.info(f"{idx}. {rec['Name']} (Confidence: {rec['confidence']:.3f})")

if __name__ == "__main__":
    main()
