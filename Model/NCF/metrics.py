import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_hit_rate(recommended_items, actual_items, k=None):
    """Calculate the fraction of recommended items that appear in the actual items."""
    if not recommended_items or not actual_items:
        logging.info(f"Empty recommended or actual items (k={k}). Returning 0.0")
        return 0.0

    k = min(k or len(recommended_items), len(recommended_items))
    hits = len(set(recommended_items[:k]) & set(actual_items)) / k
    return hits


def calculate_ndcg(recommended_items, actual_items, k):
    """Calculate Normalized Discounted Cumulative Gain (NDCG)."""
    if not recommended_items or not actual_items:
        logging.info(f"Empty recommended or actual items (k={k}). Returning 0.0")
        return 0.0

    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended_items[:k]) if item in actual_items)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual_items), k)))

    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_recall(recommended_items, actual_items, k):
    """Calculate Precision and Recall for the top-K recommended items."""
    if not recommended_items or not actual_items:
        logging.info(f"Empty recommended or actual items (k={k}). Returning (0.0, 0.0)")
        return 0.0, 0.0

    k = min(k, len(recommended_items))
    relevant = set(recommended_items[:k]) & set(actual_items)

    precision = len(relevant) / k if k > 0 else 0.0
    recall = len(relevant) / len(actual_items) if actual_items else 0.0

    return precision, recall


def calculate_rmse(predictions, targets):
    """Calculate Root Mean Squared Error (RMSE)."""
    if predictions.numel() == 0 or targets.numel() == 0:
        logging.warning("Empty predictions or targets in RMSE calculation. Returning 0.0")
        return 0.0

    mse = torch.mean((predictions - targets).cpu() ** 2).item()
    return np.sqrt(mse)


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error (MAE)."""
    if predictions.numel() == 0 or targets.numel() == 0:
        logging.warning("Empty predictions or targets in MAE calculation. Returning 0.0")
        return 0.0

    return torch.mean(torch.abs(predictions.cpu() - targets.cpu())).item()
