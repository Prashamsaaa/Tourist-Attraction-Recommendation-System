import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        # Initialize embeddings properly
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb * item_emb
        # return torch.sum(user_emb * item_emb, dim=1)
    

class MLP(nn.Module):
    def __init__(self, latent_dim, hidden_layers):
        super(MLP, self).__init__()
        input_dim = latent_dim * 2
        
        layers = []
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3 if i == 0 else 0.2))
            input_dim = units
            
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights properly
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(x)


