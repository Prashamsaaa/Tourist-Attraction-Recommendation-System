import torch
from config import *
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # # Optional: Add bias terms
        # self.user_bias = nn.Embedding(num_users, 1)
        # self.item_bias = nn.Embedding(num_items, 1)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
         
        # user_b = self.user_bias(user_ids).squeeze()
        # item_b = self.item_bias(item_ids).squeeze()
        # # Element-wise multiplication and adding biases
        # interaction = user_emb * item_emb + user_b + item_b
        # return interaction

        # Element-wise multiplication
        interaction = user_emb * item_emb
        return interaction


class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_layers):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        input_dim = latent_dim * 2
        layers = []
        for layer_size in hidden_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))
            input_dim = layer_size

        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        mlp_input = torch.cat([user_emb, item_emb], dim=-1)
        x = self.mlp_layers(mlp_input)
        return x
    # # Optional: Residual Connection
    #     return x + mlp_input


class NCF(nn.Module):
    def __init__(
        self, num_users, num_items, latent_dim=LATENT_DIM, hidden_layers=HIDDEN_LAYERS
    ):
        super(NCF, self).__init__()
        self.gmf = GMF(num_users, num_items, latent_dim)
        self.mlp = MLP(num_users, num_items, latent_dim, hidden_layers)

        fusion_dim = latent_dim + hidden_layers[-1]
        self.output_layer = nn.Sequential(
            nn.Linear(fusion_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output_layer(combined)
        # Optional: Add a non-linearity if necessary
        return torch.sigmoid(output.squeeze()) * 5.0  # Adjust scale to [0, 5]
