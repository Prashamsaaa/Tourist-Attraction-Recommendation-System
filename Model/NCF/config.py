MODEL_CONFIG = {
    "embedding_dim": 32,
    "latent_dim": 64,
    "layers": [128, 64, 32, 16],  # First layer must be bigger than 2 * latent_dim
    "dropout": 0.2,
}

TRAINING_CONFIG = {
    "batch_size": 256,
    "epochs": 20,
    "learning_rate": 0.001,
}

def validate_config():
    assert MODEL_CONFIG["embedding_dim"] > 0, "embedding_dim must be positive"
    assert MODEL_CONFIG["latent_dim"] > 0, "latent_dim must be positive"
    assert all(layer > 0 for layer in MODEL_CONFIG["layers"]), "All layers must be positive"
    assert 0 <= MODEL_CONFIG["dropout"] <= 1, "dropout must be between 0 and 1"
    assert TRAINING_CONFIG["batch_size"] > 0, "batch_size must be positive"
    assert TRAINING_CONFIG["epochs"] > 0, "epochs must be positive"
    assert TRAINING_CONFIG["learning_rate"] > 0, "learning_rate must be positive"

validate_config()
