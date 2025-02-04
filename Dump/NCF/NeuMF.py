import torch
import torch.nn as nn
from NCF.gmf_mlp import GMF
from NCF.gmf_mlp import MLP


class NCF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=64, hidden_layers=[256, 128, 64], dropout_rate=0.2):
        super(NCF, self).__init__()
        
        self.gmf = GMF(num_users, num_items, latent_dim)
        self.mlp = MLP(latent_dim, hidden_layers)
        
        # Separate embeddings for MLP
        self.mlp_user_embedding = nn.Embedding(num_users, latent_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, latent_dim)
        
        fusion_dim = latent_dim + hidden_layers[-1]
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # No activation here
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.output_layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_output = self.gmf(user_ids, item_ids)
        
        # MLP path with separate embeddings
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_output = self.mlp(mlp_user_emb, mlp_item_emb)
        
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output_layer(combined)
        # Scale the output to range [1, 5]
        rating = torch.sigmoid(output) * 4 + 1
        return rating

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            predictions = self(user_ids, item_ids)
            return predictions




