import torch

# Hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LATENT_DIM = 32
TOP_K = 5
NUM_LAYERS = 3
TEST_SIZE = 0.2
SEED = 42
HIDDEN_LAYERS = [32, 16, 8]
WEIGHT_DECAY = 0.001
# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATHS CONFIGURATION
MODEL_SAVE_PATH = "./Model/models/model.pth"
