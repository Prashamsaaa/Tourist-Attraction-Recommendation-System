# gmf_mlp.py

import torch.nn as nn
import torch
import logging

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

    def forward(self, user_ids, item_ids):
        try:
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
            return user_embeds * item_embeds  # Output shape: [batch_size, latent_dim]
        except Exception as e:
            logging.error(f"Error in GMF forward pass: {e}")
            return None

class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        # Input size is 2 * latent_dim because we concatenate user and item embeddings
        input_size = 2 * latent_dim
        
        layers = []
        layer_dims = [input_size] + hidden_layers
        
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims)-2:  # Don't add ReLU after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=0.2))
            
        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        try:
            user_embeds = self.user_embedding(user_ids)
            item_embeds = self.item_embedding(item_ids)
            x = torch.cat([user_embeds, item_embeds], dim=1)  # Changed dim=-1 to dim=1
            return self.mlp_layers(x)
        except Exception as e:
            logging.error(f"Error in MLP forward pass: {e}")
            return None
