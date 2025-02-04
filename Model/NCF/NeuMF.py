# NeuMF.py

import torch
import torch.nn as nn
import logging

class NeuMF(nn.Module):
    def __init__(self, gmf_model, mlp_model):
        super(NeuMF, self).__init__()
        self.gmf = gmf_model
        self.mlp = mlp_model
        
        # Get dimensions from models
        self.gmf_dim = gmf_model.user_embedding.embedding_dim
        
        # Find the last Linear layer's output size
        for layer in reversed(mlp_model.mlp_layers):
            if isinstance(layer, nn.Linear):
                self.mlp_dim = layer.out_features
                break
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.gmf_dim + self.mlp_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids):
        try:
            gmf_output = self.gmf(user_ids, item_ids)  # Will be [batch_size, latent_dim]
            mlp_output = self.mlp(user_ids, item_ids)  # Will be [batch_size, final_mlp_dim]
            
            # Ensure GMF output has the same dimensions as MLP output
            gmf_output = gmf_output.view(gmf_output.size(0), -1)
            
            # Now both outputs have shape [batch_size, X]
            combined = torch.cat([gmf_output, mlp_output], dim=1)
            
            # Ensure combined dimensions match the expected input for the output layer
            combined = combined.view(combined.size(0), -1)
            
            output = self.output_layer(combined)
            return output
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            return None
