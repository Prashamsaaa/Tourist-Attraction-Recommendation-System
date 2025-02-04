import torch
import numpy as np
from datetime import datetime
import json
import logging
from collections import deque

class DynamicModelManager:
    def __init__(self, model, optimizer, criterion, user_encoder, place_encoder, 
                 max_memory_size=10000, backup_frequency=100):
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
        required_fields = ['user_id', 'id', 'rating']
        
        try:
            # Check required fields
            if not all(field in rating_data for field in required_fields):
                raise ValueError(f"Missing required fields. Need: {required_fields}")
            
            # Validate rating value
            if not (0 <= rating_data['rating'] <= 5):
                raise ValueError("Rating must be between 0 and 5")
                
            return True
        except Exception as e:
            self.logger.error(f"Rating validation failed: {str(e)}")
            return False

    def save_model_state(self, path='model_backup.pt'):
        """Save current model state with metadata"""
        try:
            state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'user_encoder_classes': self.user_encoder.classes_.tolist(),
                'place_encoder_classes': self.place_encoder.classes_.tolist(),
                'version': self.current_version,
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': {
                    'recent_loss_avg': np.mean(self.recent_losses) if self.recent_losses else None
                }
            }
            torch.save(state, path)
            self.last_backup = path
            self.logger.info(f"Model state saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model state: {str(e)}")
            return False

    def load_model_state(self, path='model_backup.pt'):
        """Load model state with validation"""
        try:
            state = torch.load(path)
            
            # Validate state contents
            required_keys = ['model_state', 'optimizer_state', 'version']
            if not all(key in state for key in required_keys):
                raise ValueError("Invalid state file")
            
            # Restore state
            self.model.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optimizer_state'])
            self.user_encoder.classes_ = np.array(state['user_encoder_classes'])
            self.place_encoder.classes_ = np.array(state['place_encoder_classes'])
            self.current_version = state['version']
            
            self.logger.info(f"Model state loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model state: {str(e)}")
            return False

    def update_model(self, new_rating_data, train_dataset, train_loader):
        """Process new rating with safety checks and monitoring"""
        try:
            # Validate input
            if not self.validate_rating_data(new_rating_data):
                return False
                
            # Save state before update
            if self.update_count % self.backup_frequency == 0:
                self.save_model_state(f'model_backup_v{self.current_version}.pt')
            
            # Convert data and update model
            self.model.train()
            new_user_id = self.user_encoder.transform([new_rating_data['user_id']])[0]
            new_item_id = self.place_encoder.transform([new_rating_data['id']])[0]
            
            # Prepare tensors
            user_tensor = torch.tensor([new_user_id], dtype=torch.long).to(next(self.model.parameters()).device)
            item_tensor = torch.tensor([new_item_id], dtype=torch.long).to(next(self.model.parameters()).device)
            rating_tensor = torch.tensor([new_rating_data['rating']], dtype=torch.float32).to(next(self.model.parameters()).device)
            
            # Update step
            self.optimizer.zero_grad()
            prediction = self.model(user_tensor, item_tensor)
            loss = self.criterion(prediction.squeeze(), rating_tensor)
            loss.backward()
            self.optimizer.step()
            
            # Track performance
            self.recent_losses.append(loss.item())
            self.recent_updates.append(new_rating_data)
            self.update_count += 1
            self.current_version += 1
            
            # Log update metrics
            self.logger.info(f"Model updated - Version: {self.current_version}, Loss: {loss.item():.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")
            # Try to restore last good state
            if self.last_backup:
                self.load_model_state(self.last_backup)
            return False

    def get_model_stats(self):
        """Get current model statistics"""
        return {
            'version': self.current_version,
            'updates_processed': self.update_count,
            'avg_recent_loss': np.mean(self.recent_losses) if self.recent_losses else None,
            'user_count': len(self.user_encoder.classes_),
            'item_count': len(self.place_encoder.classes_),
            'last_backup': self.last_backup
        }
    