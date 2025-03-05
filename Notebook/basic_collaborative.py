import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import implicit
import torch

# Load user-item interaction data
df = pd.read_csv('./Model/Data/all_ratings.csv')

# Load place descriptions
description_df = pd.read_csv('./Model/Data/PreparedData.csv')

# Prepare user-item matrix and rating mask
def prepare_matrices(data):
    # Create actual ratings matrix (0 means no rating)
    user_item_matrix = data.pivot_table(
        index='user_id', 
        columns='id', 
        values='rating', 
        fill_value=0
    )
    
    # Create mask of rated items (1 if rated, 0 otherwise)
    rated_mask = data.pivot_table(
        index='user_id',
        columns='id',
        values='rating',
        aggfunc='count',
        fill_value=0
    ).values.astype(bool)
    
    return user_item_matrix, rated_mask

user_item_matrix, rated_mask = prepare_matrices(df)
user_item_matrix_sparse = scipy.sparse.csr_matrix(user_item_matrix.values)

# Create mappings for users and items
user_idx_map = {user: idx for idx, user in enumerate(user_item_matrix.index)}
item_idx_map = {item: idx for idx, item in enumerate(user_item_matrix.columns)}

# Train ALS model
model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.1,
    iterations=50
)
model.fit(user_item_matrix_sparse)

# Function to recommend places
def recommend_places(user_id, num_recommendations=5):
    if user_id not in user_idx_map:
        print(f"User '{user_id}' not found!")
        return []
    
    user_idx = user_idx_map[user_id]
    recommended_items = model.recommend(
        user_idx, 
        user_item_matrix_sparse[user_idx], 
        N=num_recommendations
    )
    
    recommendations = [(user_item_matrix.columns[item_idx], score) 
                       for item_idx, score in zip(recommended_items[0], recommended_items[1])]
    
    # Add place descriptions
    recommendations_with_descriptions = []
    for place, score in recommendations:
        desc = description_df[description_df['ID'] == place]
        if not desc.empty:
            entry = (place, desc['Name'].values[0], desc['Description'].values[0], score)
        else:
            entry = (place, 'Unknown', 'No description available', score)
        recommendations_with_descriptions.append(entry)
    
    return recommendations_with_descriptions
def calculate_metrics(actual_ratings, predicted_ratings, rated_mask, k=5, relevance_threshold=4):
    # Global error metrics using rated_mask
    actual_flat = actual_ratings[rated_mask]
    predicted_flat = predicted_ratings[rated_mask]
    
    rmse = np.sqrt(mean_squared_error(actual_flat, predicted_flat))
    mae = mean_absolute_error(actual_flat, predicted_flat)
    
    precision_scores = []
    recall_scores = []
    hit_rates = []
    ndcg_scores = []
    
    for user_idx in range(actual_ratings.shape[0]):
        user_rated = rated_mask[user_idx]
        user_actual = actual_ratings[user_idx, user_rated]
        user_predicted = predicted_ratings[user_idx, user_rated]
        
        if len(user_actual) == 0:  # Skip users with no ratings
            continue
        
        # Get top k predicted items indices
        predicted_order = np.argsort(-user_predicted)  # negative for descending order
        top_k_indices = predicted_order[:k]
        
        # Get relevant items (actual >= threshold)
        relevant_indices = np.where(user_actual >= relevance_threshold)[0]
        
        # Precision@k and Recall@k calculation
        recommended_relevant = np.intersect1d(top_k_indices, relevant_indices)
        precision = len(recommended_relevant) / k if k > 0 else 0
        recall = len(recommended_relevant) / len(relevant_indices) if len(relevant_indices) > 0 else 0
        
        # Hit Rate@k calculation - 1 if any recommended item is in actual items
        hit_rate = 1 if len(recommended_relevant) > 0 else 0
        
        # NDCG@k calculation
        dcg = 0.0
        for i, item in enumerate(top_k_indices):
            relevance = user_actual[item]  # Get rating as relevance
            dcg += relevance / np.log2(i + 2)
        
        # Calculate IDCG using actual ratings
        ideal_order = np.argsort(-user_actual)
        idcg = 0.0
        for i, item in enumerate(ideal_order[:k]):
            relevance = user_actual[item]
            idcg += relevance / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        hit_rates.append(hit_rate)
        ndcg_scores.append(ndcg)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision@k": np.mean(precision_scores),
        "Recall@k": np.mean(recall_scores),
        "HitRate@k": np.mean(hit_rates),
        "NDCG@k": np.mean(ndcg_scores)
    }

# Generate predicted ratings matrix using the trained model
predicted_ratings = model.user_factors.dot(model.item_factors.T)
actual_ratings = user_item_matrix.values

# Calculate evaluation metrics
metrics = calculate_metrics(actual_ratings, predicted_ratings, rated_mask)

# Print metrics with clear formatting
print("\nModel Evaluation Metrics:")
print(f"- RMSE: {metrics['RMSE']:.4f}")
print(f"- MAE: {metrics['MAE']:.4f}")
print(f"- Precision@5: {metrics['Precision@k']:.4f}")
print(f"- Recall@5: {metrics['Recall@k']:.4f}")
print(f"- Hit Rate@5: {metrics['HitRate@k']:.4f}")
print(f"- NDCG@5: {metrics['NDCG@k']:.4f}")

# Example usage
user_id = 1  # Replace with a valid user ID
recommendations = recommend_places(user_id)
print(f"\nTop recommendations for user {user_id}:")
for idx, rec in enumerate(recommendations, 1):
    print(f"{idx}. {rec[1]} (ID: {rec[0]}) - Score: {rec[3]:.2f}")
    print(f"   Description: {rec[2]}\n")
