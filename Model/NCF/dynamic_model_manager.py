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

        for col, encoder in [
            ("user_id", self.user_encoder),
            ("id", self.place_encoder),
        ]:
            all_classes = set(encoder.classes_)
            if new_df[col].iloc[0] not in all_classes:
                self.logger.info(
                    f"New {col} ID found: {new_df[col].iloc[0]}. Updating encoder..."
                )
                all_classes.add(new_df[col].iloc[0])
                encoder.fit(list(all_classes))
            new_df[col] = encoder.transform(new_df[col])
        return new_df

    def save_model_state(self, path="model_backup.pt"):
        """Save current model state with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%M%D_%H%M%S")
            backup_path = f"model_backup_v{self.current_version}_{timestamp}.pt"

            # Create a complete state dictionary with the entire model
            state = {
                "model_state": self.model.state_dict(),  # Save the complete model
                "optimizer_state": self.optimizer.state_dict(),  # Save the complete optimizer
                "criterion": self.criterion,  # Save the loss criterion
                "user_encoder_classes": self.user_encoder.classes_.tolist(),
                "place_encoder_classes": self.place_encoder.classes_.tolist(),
                "version": self.current_version,
                "update_count": self.update_count,
                "timestamp": datetime.now().isoformat(),
            }

            torch.save(state, backup_path)
            self.last_backup = backup_path
            self.logger.info(f"Complete model state saved to {backup_path}")
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

    def update_model(
        self, new_rating_data, train_dataset, device, retrain_threshold=100
    ):
        """Update model with new rating data and optionally retrain the model"""
        try:
            if not self.validate_rating_data(new_rating_data):
                return False

            updated_df = self.update_encoders(new_rating_data)
            if updated_df is None:
                return False

            # Step 1: Add new data to the temporary buffer
            new_user_id = torch.tensor(
                updated_df["user_id"].values, dtype=torch.long
            ).to(device)
            new_item_id = torch.tensor(updated_df["id"].values, dtype=torch.long).to(
                device
            )
            new_rating = torch.tensor(
                updated_df["rating"].values, dtype=torch.float32
            ).to(device)

            # Add the new data to the recent updates buffer
            self.recent_updates.append((new_user_id, new_item_id, new_rating))

            # Step 2: Process in mini-batches if buffer is full
            if len(self.recent_updates) >= self.batch_size:
                # Prepare the mini-batch
                batch = list(self.recent_updates)
                self.recent_updates.clear()  # Reset the buffer

                batch_users, batch_items, batch_ratings = zip(*batch)

                # Stack tensors to create a mini-batch
                batch_users = torch.stack(batch_users).to(device)
                batch_items = torch.stack(batch_items).to(device)
                batch_ratings = torch.stack(batch_ratings).to(device)

                # Step 3: Perform a training step on the mini-batch
                self.optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = self.criterion(predictions.squeeze(), batch_ratings)
                loss.backward()
                self.optimizer.step()

                # Step 4: Track the losses and updates
                self.recent_losses.append(loss.item())
                self.update_count += 1
                self.current_version += 1

                self.logger.info(
                    f"Model updated - Version: {self.current_version}, Loss: {loss.item():.4f}"
                )

                # Step 5: Check if retraining is necessary
                if self.update_count % retrain_threshold == 0:
                    self.retrain_model(train_dataset, device)
                    self.logger.info(
                        f"Model retrained after {self.update_count} updates."
                    )

            else:
                # Log the update without training if buffer is not full
                self.logger.info(f"Added to buffer: {len(self.recent_updates)} samples")

            return True

        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")
            if self.last_backup:
                self.load_model_state(self.last_backup)
            return False

    def retrain_model(self, train_dataset, device):
        """Retrain the model with the full dataset"""
        try:
            self.logger.info("Starting full model retraining...")

            # Reset model, optimizer, and loss history if needed
            self.model.reset_parameters()  # Assuming you have a reset method
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

            # Training loop for full retraining
            for epoch in range(self.num_epochs):
                self.model.train()
                for user_ids, item_ids, ratings in train_dataset:
                    user_ids = user_ids.to(device)
                    item_ids = item_ids.to(device)
                    ratings = ratings.to(device)

                    self.optimizer.zero_grad()
                    predictions = self.model(user_ids, item_ids)
                    loss = self.criterion(predictions.squeeze(), ratings)
                    loss.backward()
                    self.optimizer.step()

                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} - Loss: {loss.item():.4f}"
                )

            # Save the retrained model
            self.save_model_state(f"retrained_model_v{self.current_version}.pt")

        except Exception as e:
            self.logger.error(f"Retraining failed: {str(e)}")
            if self.last_backup:
                self.load_model_state(self.last_backup)

    def get_model_stats(self):
        """Get current model statistics"""
        return {
            "version": self.current_version,
            "updates_processed": self.update_count,
            "avg_recent_loss": (
                np.mean(self.recent_losses) if self.recent_losses else 0.0
            ),
            "user_count": len(self.user_encoder.classes_),
            "item_count": len(self.place_encoder.classes_),
            "last_backup": self.last_backup,
        }
