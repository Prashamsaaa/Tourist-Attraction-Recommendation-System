from NCF.NeuMF import NCF
import torch

class Recommender:
    def __init__(self, model_path, item_mapping):
        self.model_path = model_path
        self.item_mapping = item_mapping
        # Load state dict first to get dimensions
        self.state_dict = torch.load(self.model_path)
        self.load_model()

    def load_model(self):
        # Get dimensions from the pre-trained model
        pretrained_item_dim = self.state_dict['gmf.item_embedding.weight'].shape[0]
        latent_dim = self.state_dict['gmf.item_embedding.weight'].shape[1]
        
        # Initialize model with same dimensions as pre-trained model
        self.model = NCF(
            num_users=10,  # Fixed number of users
            num_items=pretrained_item_dim,  # Use pre-trained dimension
            latent_dim=latent_dim,
            hidden_layers=[64, 32]
        )
        
        # Rename keys to match model's state dict
        new_state_dict = {}
        for key, value in self.state_dict.items():
            if key.startswith("user_embedding"):
                new_key = key.replace("user_embedding", "gmf.user_embedding")
            elif key.startswith("item_embedding"):
                new_key = key.replace("item_embedding", "gmf.item_embedding")
            else:
                new_key = key
            new_state_dict[new_key] = value

        # Load the state dict
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

    def recommend(self, user_id, top_n=5, valid_item_ids=None):
        """
        Generate recommendations for a user, optionally filtered by valid_item_ids (e.g. for province).
        """
        try:
            with torch.no_grad():
                # Map item IDs to pre-trained model's item space
                available_items = torch.tensor(range(self.state_dict['gmf.item_embedding.weight'].shape[0]))
                user_tensor = torch.tensor([user_id] * len(available_items))

                # Get predictions
                predictions = self.model(user_tensor, available_items)

                # Sort and pick top N
                top_indices = predictions.squeeze().argsort(descending=True)
                recommended_items = [(idx.item(), predictions[idx].item()) for idx in top_indices]

                # Filter by valid_item_ids if supplied
                if valid_item_ids is not None:
                    recommended_items = [r for r in recommended_items if r[0] in valid_item_ids]

                # Slice top N from filtered results
                recommended_items = recommended_items[:top_n]
                return recommended_items

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
