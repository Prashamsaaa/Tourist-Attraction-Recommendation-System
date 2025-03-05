import torch
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime
import torch.nn as nn

class DynamicModelManager:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        user_encoder,
        place_encoder,
        max_memory_size=10000,
        backup_frequency=100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.user_encoder = user_encoder
        self.place_encoder = place_encoder

        # State management
        self.current_version = 0
        self.update_count = 0
        self.last_backup = None
        self.backup_frequency = backup_frequency

        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.recent_losses = deque(maxlen=10)

        # Memory management
        self.max_memory_size = max_memory_size
        self.recent_updates = deque(maxlen=max_memory_size)

        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def validate_rating_data(self, rating_data):
        """Validate new rating data"""
        required_fields = ["user_id", "id", "rating"]

        try:
            if not all(field in rating_data for field in required_fields):
                raise ValueError(f"Missing required fields. Need: {required_fields}")

            if not (0 <= rating_data["rating"] <= 5):
                raise ValueError("Rating must be between 0 and 5")

            return True
        except Exception as e:
            self.logger.error(f"Rating validation failed: {str(e)}")
            return False

    def update_encoders(self, new_rating_data):
        """Update encoders with new user/item IDs if needed"""
        new_df = pd.DataFrame([new_rating_data])

        for col, encoder in [
            ("user_id", self.user_encoder),
            ("id", self.place_encoder),
        ]:
            original_id = new_df[col].iloc[0]
            
            # Dynamically expand encoder if new ID is encountered
            if original_id not in encoder.classes_:
                self.logger.info(
                    f"New {col.replace('_', ' ').title()} found: {original_id}. Adding to encoder."
                )
                encoder.classes_ = np.append(encoder.classes_, original_id)
                
                # Resize model's embedding layer to accommodate new entities
                if col == "user_id":
                    self._resize_embedding_layer(self.model.user_embedding, len(encoder.classes_))
                else:
                    self._resize_embedding_layer(self.model.item_embedding, len(encoder.classes_))
            
            # Transform to encoded value
            try:
                new_df[col] = encoder.transform([original_id])[0]
            except Exception as e:
                self.logger.error(f"Error encoding {col}: {e}")
                return None

        return new_df

    def _resize_embedding_layer(self, embedding_layer, new_num_embeddings):
        """Safely resize an embedding layer"""
        old_embedding = embedding_layer
        new_embedding = nn.Embedding(new_num_embeddings, embedding_layer.embedding_dim)
        
        # Copy existing weights
        new_embedding.weight.data[:old_embedding.num_embeddings] = old_embedding.weight.data
        
        # Initialize new weights with Xavier uniform 
        nn.init.xavier_uniform_(new_embedding.weight[old_embedding.num_embeddings:])
        
        # Replace the old embedding with the new one
        embedding_layer = new_embedding

def save_model_state(self, path="model_backup.pt"):
    """Save current model state with metadata"""
    try:
        state = {
            "model_state_dict": self.model.state_dict(),  # Save only state dict
            "optimizer_state_dict": self.optimizer.state_dict(),
            "user_encoder_classes": self.user_encoder.classes_.tolist(),
            "place_encoder_classes": self.place_encoder.classes_.tolist(),
            "version": self.current_version,
            "update_count": self.update_count,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {
                "recent_loss_avg": (
                    np.mean(self.recent_losses) if self.recent_losses else None
                ),
                "recent_losses": list(self.recent_losses),
                "performance_history": list(self.performance_history),
            },
        }

        torch.save(state, path)
        self.last_backup = path
        self.logger.info(f"Model state saved to {path}")
        return True

    except Exception as e:
        self.logger.error(f"Failed to save model state: {str(e)}")
        return False
def load_model_state(self, path="model_backup.pt"):
    """Load model state from file"""
    try:
        state = torch.load(path, map_location=torch.device("cpu"))

        # Load model and optimizer states
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        # Restore encoder classes
        self.user_encoder.classes_ = np.array(state["user_encoder_classes"])
        self.place_encoder.classes_ = np.array(state["place_encoder_classes"])

        # Restore version and counters
        self.current_version = state["version"]
        self.update_count = state["update_count"]

        # Restore performance metrics
        if "performance_metrics" in state:
            metrics = state["performance_metrics"]
            self.recent_losses = deque(
                metrics.get("recent_losses", []), 
                maxlen=10
            )
            self.performance_history = deque(
                metrics.get("performance_history", []), 
                maxlen=100
            )

        self.logger.info(f"Model state loaded from {path}")
        return True

    except Exception as e:
        self.logger.error(f"Failed to load model state: {str(e)}")
        return False
    def update_model(self, new_rating_data, train_dataset, device):
        """Update model with new rating data"""
        try:
            # Validate input data
            if not self.validate_rating_data(new_rating_data):
                return False

            # Update encoders and get encoded data
            updated_df = self.update_encoders(new_rating_data)
            if updated_df is None:
                return False

            # Periodic backup
            if self.update_count % self.backup_frequency == 0:
                self.save_model_state(f"model_backup_v{self.current_version}.pt")

            # Prepare tensors
            new_user_id = torch.tensor(
                updated_df["user_id"].values, 
                dtype=torch.long, 
                device=device
            )
            new_item_id = torch.tensor(
                updated_df["id"].values, 
                dtype=torch.long, 
                device=device
            )
            new_rating = torch.tensor(
                updated_df["rating"].values, 
                dtype=torch.float32, 
                device=device
            )

            # Update training dataset
            train_dataset.user_ids = torch.cat([
                train_dataset.user_ids.to(device), 
                new_user_id
            ])
            train_dataset.item_ids = torch.cat([
                train_dataset.item_ids.to(device), 
                new_item_id
            ])
            train_dataset.ratings = torch.cat([
                train_dataset.ratings.to(device), 
                new_rating
            ])

            # Perform model update
            self.optimizer.zero_grad()
            self.model.eval()  # Evaluation mode for prediction
            
            # Ensure model can handle new indices
            predictions = self.model(new_user_id, new_item_id)
            loss = self.criterion(predictions.squeeze(), new_rating)
            
            self.model.train()  # Back to training mode
            loss.backward()
            self.optimizer.step()

            # Update tracking
            self.recent_losses.append(loss.item())
            self.update_count += 1
            self.current_version += 1

            self.logger.info(
                f"Model updated - Version: {self.current_version}, Loss: {loss.item():.4f}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}", exc_info=True)
            
            # Attempt to restore from last backup if available
            if self.last_backup:
                self.load_model_state(self.last_backup)
            
            return False

    def get_model_stats(self):
        """Get current model statistics"""
        return {
            "version": self.current_version,
            "updates_processed": self.update_count,
            "avg_recent_loss": (
                np.mean(self.recent_losses) if self.recent_losses else None
            ),
            "user_count": len(self.user_encoder.classes_),
            "item_count": len(self.place_encoder.classes_),
            "last_backup": self.last_backup,
        }