import torch
import json

# Hyperparameters
NUM_EPOCHS = 20

BATCH_SIZE = 32
LEARNING_RATE = 0.01
LATENT_DIM = 64

TOP_K = 5
NUM_LAYERS = 3
TEST_SIZE = 0.2
SEED = 42
HIDDEN_LAYERS = [64, 32, 16]
WEIGHT_DECAY = 0.1
DROPOUT_RATE = 0.7

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS CONFIGURATION
MODEL_SAVE_PATH = "./Model/models/model.pth"

CONFIG_PATH = "config.json"
def save_config(last_epoch_metrics):
    config = {
        "last_epoch_metrics": last_epoch_metrics
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config saved at {CONFIG_PATH}")