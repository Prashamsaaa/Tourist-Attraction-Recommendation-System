import numpy as np
import torch


def calculate_hit_rate(recommended_items, actual_items, k=None):
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    # Handle edge case when no recommended items or actual items
    if not recommended_items or not actual_items:
        return 0

    hit = len(set(recommended_items[:k]) & set(actual_items)) > 0
    return int(hit)


def calculate_ndcg(recommended_items, actual_items, ratings, k):
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

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
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]

    k = min(
        k, len(recommended_items)
    )  # Ensure k does not exceed length of recommended items
    relevant = set(recommended_items[:k]) & set(actual_items)
    precision = len(relevant) / k if k > 0 else 0
    recall = len(relevant) / len(actual_items) if actual_items else 0
    return precision, recall


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error"""
    if predictions.size(0) == 0 or targets.size(0) == 0:
        return float("nan")  # or return 0.0 if that fits your use case

    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error"""
    if predictions.size(0) == 0 or targets.size(0) == 0:
        return float("nan")

    return torch.mean(torch.abs(predictions - targets)).item()
