import torch
import json

# Hyperparameters
NUM_EPOCHS = 30

BATCH_SIZE = 128
LEARNING_RATE = 0.0005
LATENT_DIM = 128

TOP_K = 5
NUM_LAYERS = 3
TEST_SIZE = 0.2
SEED = 32
HIDDEN_LAYERS = [128, 64, 32, 16]

WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.2

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS CONFIGURATION
MODEL_SAVE_PATH = "./Model/models/model.pth"
