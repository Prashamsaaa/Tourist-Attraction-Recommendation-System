import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb * item_emb

class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        input_dim = latent_dim * 2
        layers = []
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))
            input_dim = layer_size

        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        mlp_input = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp_layers(mlp_input)

class NCF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=64, hidden_layers=[256, 128, 64]):
        super(NCF, self).__init__()
        self.gmf = GMF(num_users, num_items, latent_dim)
        self.mlp = MLP(num_users, num_items, latent_dim, hidden_layers)
        
        fusion_dim = latent_dim + hidden_layers[-1]
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output_layer(combined)
        return output

    def predict(self, user_ids, item_ids):
        with torch.no_grad():
            return self(user_ids, item_ids)
