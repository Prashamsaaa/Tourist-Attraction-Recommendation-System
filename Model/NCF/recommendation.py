
import torch

def generate_recommendations(model, user_id, num_items=None, top_k=10, device=None):
    """
    Generate top-k recommendations for a given user.
    """
    if num_items is None:
        raise ValueError("num_items must be provided.")

    item_ids = torch.arange(num_items).to(device)  # Create a tensor of item IDs
    user_ids = torch.tensor([user_id] * num_items).to(device)  # Repeat the user ID

    # Predict scores for all items for this user
    scores = model(user_ids, item_ids).squeeze()

    # Get the top-k item indices based on scores
    _, top_k_indices = torch.topk(scores, top_k)

    return top_k_indices.cpu().numpy()  # Return top-k item IDs as a numpy array
