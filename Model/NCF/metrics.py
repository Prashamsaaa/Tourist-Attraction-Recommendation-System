import numpy as np
import torch

def calculate_hit_rate(recommended_items, actual_items):
    """Calculate if any recommended item is in the actual items"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
    return int(len(set(recommended_items) & set(actual_items)) > 0)

def calculate_ndcg(recommended_items, actual_items, k):
    """Calculate Normalized Discounted Cumulative Gain"""
    if isinstance(actual_items, (int, np.integer)):
        actual_items = [int(actual_items)]
        
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in actual_items:
            dcg += 1 / np.log2(i + 2)
    
    idcg = 1.0
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
