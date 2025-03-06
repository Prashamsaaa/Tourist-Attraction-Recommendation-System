import torch
import pandas as pd
import numpy as np
import logging
from collections import deque
from datetime import datetime

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
        logging.basicConfig(level=logging.INFO)
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

        user_id = new_rating_data['user_id']
        item_id = new_rating_data['id']

        # Update user encoder
        if user_id not in self.user_encoder.classes_:
            self.logger.info(f"New User ID found: {user_id}. Adding to encoder.")
            self.user_encoder.classes_ = np.append(self.user_encoder.classes_, user_id)

        # Update item encoder
        if item_id not in self.place_encoder.classes_:
            self.logger.info(f"New Item ID found: {item_id}. Adding to encoder.")
            self.place_encoder.classes_ = np.append(self.place_encoder.classes_, item_id)

        # Transform the user and item IDs
        encoded_user_id = self.user_encoder.transform([user_id])[0]
        encoded_item_id = self.place_encoder.transform([item_id])[0]

        return encoded_user_id, encoded_item_id

    def save_model_state(self, path="model_backup.pt", save_dataset=True):
        """Save current model state with metadata and optionally the dataset"""
        try:
            # Create a complete state dictionary with the entire model
            state = {
                "model": self.model,  # Save the complete model
                "optimizer": self.optimizer,  # Save the complete optimizer
                "criterion": self.criterion,  # Save the loss criterion
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
            self.logger.info(f"Complete model state saved to {path}")

            if save_dataset:
                # Convert dataset to DataFrame and save to CSV
                try:
                    df = pd.DataFrame({
                        'user_id': [self.user_encoder.inverse_transform([int(user_id)])[0] for user_id in self.train_dataset.user_ids.cpu().numpy()],
                        'item_id': [self.place_encoder.inverse_transform([int(item_id)])[0] for item_id in self.train_dataset.item_ids.cpu().numpy()],
                        'rating': self.train_dataset.ratings.cpu().numpy()
                    })
                    dataset_path = "updated_dataset.csv"
                    df.to_csv(dataset_path, index=False)
                    self.logger.info(f"Updated dataset saved to {dataset_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save dataset: {str(e)}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save model state: {str(e)}")
            return False


    def load_model_state(self, path="model_backup.pt"):
        """Load complete model state from file"""
        try:
            state = torch.load(path, map_location=torch.device("cpu"))

            # Load complete components
            self.model = state["model"]
            self.optimizer = state["optimizer"]
            self.criterion = state["criterion"]

            # Load encoder data
            self.user_encoder.classes_ = np.array(state["user_encoder_classes"])
            self.place_encoder.classes_ = np.array(state["place_encoder_classes"])

            # Load version and counters
            self.current_version = state["version"]
            self.update_count = state["update_count"]

            # Load performance metrics if available
            if "performance_metrics" in state:
                metrics = state["performance_metrics"]
                self.recent_losses = deque(metrics["recent_losses"], maxlen=10)
                self.performance_history = deque(
                    metrics["performance_history"], maxlen=100
                )

            self.logger.info(f"Complete model state loaded from {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model state: {str(e)}")
            return False

    # def load_model_state(self, path='model_backup.pt'):
    #     """Load model state from file"""
    #     try:
    #         state = torch.load(path)
    #         required_keys = ['model_state', 'optimizer_state', 'version']
    #         if not all(key in state for key in required_keys):
    #             raise ValueError("Invalid state file")

    #         self.model.load_state_dict(state['model_state'])
    #         self.optimizer.load_state_dict(state['optimizer_state'])
    #         self.user_encoder.classes_ = np.array(state['user_encoder_classes'])
    #         self.place_encoder.classes_ = np.array(state['place_encoder_classes'])
    #         self.current_version = state['version']
    #         self.logger.info(f"Model state loaded from {path}")
    #         return True
    #     except Exception as e:
    #         self.logger.error(f"Failed to load model state: {str(e)}")
    #         return False

    def update_model(self, new_rating_data, train_dataset, device):
        """Update model with new rating data"""
        try:
            if not self.validate_rating_data(new_rating_data):
                return False

            # Update encoders and get encoded IDs
            encoded_user_id, encoded_item_id = self.update_encoders(new_rating_data)

            new_user_id = torch.tensor([encoded_user_id], dtype=torch.long, device=device)
            new_item_id = torch.tensor([encoded_item_id], dtype=torch.long, device=device)
            new_rating = torch.tensor([new_rating_data['rating']], dtype=torch.float32, device=device)

            # Update training dataset
            train_dataset.user_ids = torch.cat(
                (train_dataset.user_ids.to(device), new_user_id)
            )
            train_dataset.item_ids = torch.cat(
                (train_dataset.item_ids.to(device), new_item_id)
            )
            train_dataset.ratings = torch.cat(
                (train_dataset.ratings.to(device), new_rating)
            )

            # Assign the train_dataset to self so that it can be accessed in save_model_state
            self.train_dataset = train_dataset

            # Training step
            self.optimizer.zero_grad()
            self.model.eval()  # Set model to evaluation mode
            predictions = self.model(new_user_id, new_item_id)
            loss = self.criterion(predictions.squeeze(), new_rating)
            self.model.train() # Set model back to training mode
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
            self.logger.error(f"Model update failed: {str(e)}")
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
