import numpy as np
import torch


def calculate_hit_rate(recommended_items, actual_items, k=None):
    """Calculate the fraction of recommended items that are in the actual items."""
    
    # Convert actual_items to a list if it's a single integer
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
    
    # Convert NumPy arrays to lists if needed
    if isinstance(recommended_items, np.ndarray):
        recommended_items = recommended_items.tolist()
    if isinstance(actual_items, np.ndarray):
        actual_items = actual_items.tolist()

    # Handle edge case when no recommended items or actual items
    if not recommended_items or not actual_items:
        return 0.0
    print("Recommended items:",recommended_items)
    print("Actual items:", actual_items)

    # Ensure k is valid
    if k is None or k > len(recommended_items):
        k = len(recommended_items)

    # Compute the number of hits (items in both recommended and actual sets)
    hits = len(set(recommended_items[:k]) & set(actual_items))

    # Avoid division by zero
    return hits / k if k > 0 else 0.0


def calculate_ndcg(recommended_items, actual_items, ratings, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG)"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    # Convert NumPy arrays to lists if needed
    if isinstance(recommended_items, np.ndarray):
        recommended_items = recommended_items.tolist()
    if isinstance(actual_items, np.ndarray):
        actual_items = actual_items.tolist()

    # Relevance score based on ratings: higher rating means more relevant
    actual_relevance = {item: rating for item, rating in zip(actual_items, ratings)}

    # Ideal DCG assumes the recommended items are sorted by relevance
    ideal_relevance = sorted(ratings, reverse=True)
    idcg = sum(
        relevance / np.log2(i + 2) for i, relevance in enumerate(ideal_relevance[:k])
    )

    dcg = sum(
        actual_relevance.get(item, 0) / np.log2(i + 2)
        for i, item in enumerate(recommended_items[:k])
    )

    return dcg / idcg if idcg > 0 else 0


def calculate_precision_recall(recommended_items, actual_items, k):
    """Calculate Precision and Recall for the top-K recommended items."""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    # Convert NumPy arrays to lists if needed
    if isinstance(recommended_items, np.ndarray):
        recommended_items = recommended_items.tolist()
    if isinstance(actual_items, np.ndarray):
        actual_items = actual_items.tolist()

    k = min(k, len(recommended_items))  # Ensure k does not exceed length of recommended items
    relevant = set(recommended_items[:k]) & set(actual_items)

    # Precision: fraction of recommended items that are relevant
    precision = len(relevant) / k if k > 0 else 0

    # Recall: fraction of relevant items that are recommended
    recall = len(relevant) / len(actual_items) if actual_items else 0

    return precision, recall


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error"""
    if predictions.size(0) == 0 or targets.size(0) == 0:
        return float("nan")  # or return 0.0 if that fits your use case

    mse = torch.mean((predictions - targets).cpu() ** 2).item()
    return np.sqrt(mse)


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error"""
    if predictions.size(0) == 0 or targets.size(0) == 0:
        return float("nan")

    mae = torch.mean(torch.abs(predictions.cpu() - targets.cpu())).item()
    return mae
