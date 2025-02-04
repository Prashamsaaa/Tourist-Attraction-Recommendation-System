# Configuration parameters
class Config:
    # Data parameters
    DATA_PATH = "../Data/FinalDataset/all_ratings.csv"
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Model parameters
    LATENT_DIM = 8
    NUM_LAYERS = 3
    DROPOUT_RATE = 0.2
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING = 10
