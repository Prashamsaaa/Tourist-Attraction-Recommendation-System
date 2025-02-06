import numpy as np
import torch


def calculate_hit_rate(recommended_items, actual_items, k=None):
    """Calculate if any recommended item is in the actual items"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    hit = len(set(recommended_items[:k]) & set(actual_items)) > 0
    return int(hit)



def calculate_ndcg(recommended_items, actual_items, ratings, k):
    """Calculate Normalized Discounted Cumulative Gain with rating-based relevance"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    # Relevance score based on ratings: higher rating means more relevant
    actual_relevance = {
        item: rating for item, rating in zip(actual_items, ratings)
    }  # map item to its rating

    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        # Relevance based on rating for the item
        relevance = actual_relevance.get(
            item, 0
        )  # default relevance 0 if item not in actual items
        dcg += relevance / np.log2(i + 2)

    # Ideal DCG (IDCG) assumes the recommended items are sorted by relevance
    ideal_relevance = sorted(
        ratings, reverse=True
    )  # Ideal relevance is highest ratings first
    idcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        relevance = actual_relevance.get(item, 0)
        idcg += relevance / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0


def calculate_precision_recall(recommended_items, actual_items, k):
    """Calculate Precision@K and Recall@K"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    relevant = set(recommended_items[:k]) & set(actual_items)
    precision = len(relevant) / k if k > 0 else 0
    recall = len(relevant) / len(actual_items) if actual_items else 0
    return precision, recall


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error"""
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(predictions - targets)).item()
