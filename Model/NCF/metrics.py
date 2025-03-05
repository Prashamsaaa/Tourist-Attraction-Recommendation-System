import numpy as np
import torch

def calculate_hit_rate(recommended_items, actual_items, k=None):
    """
    Calculate hit rate for recommended items.
    
    Args:
        recommended_items (list): List of recommended item IDs
        actual_items (list or int): Actual item(s) the user likes
        k (int, optional): Top-k recommendations to consider. If None, use full list.
    
    Returns:
        int: 1 if hit, 0 otherwise
    """
    # Convert single item to list if needed
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
    
    # Use k if specified, otherwise use full list of recommended items
    k = k if k is not None else len(recommended_items)
    
    # Check if any recommended item is in the actual items
    hit = len(set(recommended_items[:k]) & set(actual_items)) > 0
    return int(hit)

def calculate_ndcg(recommended_items, actual_items, ratings, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG)
    
    Args:
        recommended_items (list): List of recommended item IDs
        actual_items (list or int): Actual item(s) the user likes
        ratings (list): Ratings corresponding to actual items (0-5 scale)
        k (int): Top-k recommendations to consider
    
    Returns:
        float: NDCG score between 0 and 1
    """
    # Convert single item to list if needed
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
    
    # Ensure ratings match actual_items length
    if len(ratings) != len(actual_items):
        raise ValueError("Length of ratings must match length of actual_items")
    
    # Map items to their ratings (0-5 scale)
    actual_relevance = {
        item: rating for item, rating in zip(actual_items, ratings)
    }
    
    # Calculate Discounted Cumulative Gain (DCG)
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        # Get relevance, default to 0 if item not in actual items
        relevance = actual_relevance.get(item, 0)
        # Use standard DCG formula with log base 2 discount
        dcg += (2**relevance - 1) / np.log2(i + 2)
    
    # Calculate Ideal DCG (IDCG)
    # Sort ratings in descending order
    sorted_ratings = sorted(ratings, reverse=True)
    
    idcg = 0.0
    for i, rating in enumerate(sorted_ratings[:k]):
        # Use standard DCG formula with log base 2 discount
        idcg += (2**rating - 1) / np.log2(i + 2)
    
    # Normalize DCG by IDCG
    return dcg / idcg if idcg > 0 else 0

def calculate_precision_recall(recommended_items, actual_items, k):
    """
    Calculate Precision@K and Recall@K
    
    Args:
        recommended_items (list): List of recommended item IDs
        actual_items (list or int): Actual item(s) the user likes
        k (int): Top-k recommendations to consider
    
    Returns:
        tuple: (precision, recall)
    """
    # Convert single item to list if needed
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
    
    # Find relevant items in top-k recommendations
    relevant = set(recommended_items[:k]) & set(actual_items)
    
    # Calculate precision (proportion of recommended items that are relevant)
    precision = len(relevant) / k if k > 0 else 0
    
    # Calculate recall (proportion of relevant items found)
    recall = len(relevant) / len(actual_items) if actual_items else 0
    
    return precision, recall

def calculate_rmse(predictions, targets):
    """
    Calculate Root Mean Squared Error
    
    Args:
        predictions (torch.Tensor): Predicted ratings
        targets (torch.Tensor): Actual ratings
    
    Returns:
        float: RMSE value
    """
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def calculate_mae(predictions, targets):
    """
    Calculate Mean Absolute Error
    
    Args:
        predictions (torch.Tensor): Predicted ratings
        targets (torch.Tensor): Actual ratings
    
    Returns:
        float: MAE value
    """
    return torch.mean(torch.abs(predictions - targets)).item()