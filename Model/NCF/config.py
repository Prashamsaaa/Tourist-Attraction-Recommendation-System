import torch

# Hyperparameters
NUM_EPOCHS = 20

BATCH_SIZE = 128
LEARNING_RATE = 0.001
LATENT_DIM = 32

TOP_K = 10
NUM_LAYERS = 3
TEST_SIZE = 0.2
SEED = 43
HIDDEN_LAYERS = [32, 16, 8]
WEIGHT_DECAY = 0.1
DROPOUT_RATE = 0.4

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS CONFIGURATION
MODEL_SAVE_PATH = "./Model/models/model.pth"
