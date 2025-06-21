import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

# Step 1: Load Data
def load_data(description_file, rating_file):
    descriptions = pd.read_csv(description_file)
    ratings = pd.read_csv(rating_file)
    return descriptions, ratings

# Step 2: Load DistilBERT Model and Tokenizer
def load_distilbert_model():
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to('cuda')
    return tokenizer, model

# Step 3: Generate Embeddings for Descriptions
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def generate_all_embeddings(data, tokenizer, model):
    embeddings = []
    for description in data['Description']:
        embeddings.append(generate_embedding(description, tokenizer, model))
    return embeddings

# Step 4: Recommend Places for a User
def recommend_places(user_id, data, ratings, embeddings, top_n=5):
    user_ratings = ratings[ratings['user_id'] == user_id]
    
    if user_ratings.empty:
        print(f"No ratings found for user {user_id}.")
        return pd.DataFrame()
    
    # Calculate similarity scores for all places based on embeddings
    recommendations = []
    for _, row in user_ratings.iterrows():
        place_id = row['id']
        place_embedding = embeddings[place_id - 1].reshape(1, -1)
        similarities = cosine_similarity(place_embedding, embeddings).flatten()
        
        # Add similarity scores to the dataset
        data['Similarity_Score'] = similarities
        
        # Compute recommendation score (weighted combination of similarity and user rating)
        data['Recommendation_Score'] = 0.7 * data['Similarity_Score'] + 0.3 * row['rating']
        
        # Get top N recommendations
        recommended_places = data.nlargest(top_n, 'Recommendation_Score')
        recommendations.append(recommended_places)
    
    # Combine recommendations and remove duplicates
    final_recommendations = pd.concat(recommendations).drop_duplicates(subset=['ID']).nlargest(top_n, 'Recommendation_Score')
    
    return final_recommendations[['ID', 'Name', 'Description', 'Province', 'Tags', 'Recommendation_Score']]

# Step 5: Evaluation Metrics

RELEVANCE_THRESHOLD = 4

# Proportional Hit Rate at K
def proportional_hit_rate_at_k(recommended_places, relevant_places, k):
    recommended_set = set(recommended_places[:k])
    relevant_set = set(relevant_places)
    
    hits = len(recommended_set & relevant_set)  # Intersection of recommended and relevant places
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

# Precision at K
def precision_at_k(recommended_places, relevant_places, k):
    recommended_set = set(recommended_places[:k])
    relevant_set = set(relevant_places)
    
    hits = len(recommended_set & relevant_set)  # Intersection of recommended and relevant places
    return hits / k if k > 0 else 0.0

# Recall at K
def recall_at_k(recommended_places, relevant_places, k):
    recommended_set = set(recommended_places[:k])
    relevant_set = set(relevant_places)
    
    hits = len(recommended_set & relevant_set)  # Intersection of recommended and relevant places
    return hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

# NDCG at K
def ndcg_at_k(recommended_places, relevant_places, k):
    dcg = 0.0
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant_places), k))])
    
    for idx, item in enumerate(recommended_places[:k]):
        if item in relevant_places:
            dcg += 1.0 / np.log2(idx + 2)

    return dcg / idcg if idcg > 0 else 0.0

# RMSE (Root Mean Square Error)
def rmse(recommended_place_ids, relevant_place_ids, ratings_df):
    relevant_ratings = ratings_df.loc[ratings_df['id'].isin(relevant_place_ids), 'rating'].values
    predicted_ratings = [5] * len(relevant_place_ids)  # Placeholder predicted rating
    
    if len(predicted_ratings) != len(relevant_ratings) or len(relevant_ratings) == 0:
        return 0.0
    
    squared_errors = (np.array(predicted_ratings) - np.array(relevant_ratings)) ** 2
    return np.sqrt(np.mean(squared_errors))

def mae(recommended_place_ids, relevant_place_ids, ratings_df):
    # Get actual ratings for relevant places
    relevant_ratings = ratings_df.loc[ratings_df['id'].isin(relevant_place_ids), 'rating'].values
    
    # Create placeholder predictions (same as RMSE example)
    predicted_ratings = [5] * len(relevant_place_ids)  # Placeholder predicted rating
    
    # Handle edge cases
    if len(predicted_ratings) != len(relevant_ratings) or len(relevant_ratings) == 0:
        return 0.0
    
    # Calculate absolute errors and mean
    absolute_errors = np.abs(np.array(predicted_ratings) - np.array(relevant_ratings))
    return np.mean(absolute_errors)

# Evaluate a Single User
def evaluate_user(user_id, data, ratings, embeddings, k=5):
    test_user_ratings = ratings[ratings['user_id'] == user_id]
    
    relevant_places = test_user_ratings[test_user_ratings['rating'] >= RELEVANCE_THRESHOLD]['id'].tolist()
    
    if not relevant_places:
        return None
    
    recommendations = recommend_places(user_id, data.copy(), ratings.copy(), embeddings)['ID'].tolist()
    
    proportional_hit_rate = proportional_hit_rate_at_k(recommendations, relevant_places, k)
    precision_score = precision_at_k(recommendations, relevant_places, k)
    recall_score = recall_at_k(recommendations, relevant_places, k)
    ndcg_score = ndcg_at_k(recommendations, relevant_places, k)
    
    rmse_score = rmse(recommendations[:k], relevant_places[:k], test_user_ratings)
    mae_score = mae(recommendations[:k], relevant_places[:k], test_user_ratings)
    
    return {'proportional_hit_rate': proportional_hit_rate,
            'precision': precision_score,
            'recall': recall_score,
            'ndcg': ndcg_score,
            'rmse': rmse_score,
            'mae': mae_score}

# Evaluate All Users
def evaluate_system(data, ratings, embeddings):
    users = ratings['user_id'].unique()
    
    metrics = {'proportional_hit_rate': [], 'precision': [], 'recall': [], 'ndcg': [], 'rmse': [], 'mae':[]}
    
    for user_id in users:
        user_metrics = evaluate_user(user_id, data.copy(), ratings.copy(), embeddings)
        
        if user_metrics:
            metrics['proportional_hit_rate'].append(user_metrics['proportional_hit_rate'])
            metrics['precision'].append(user_metrics['precision'])
            metrics['recall'].append(user_metrics['recall'])
            metrics['ndcg'].append(user_metrics['ndcg'])
            metrics['rmse'].append(user_metrics['rmse'])
            metrics['mae'].append(user_metrics['mae'])
    
    avg_proportional_hit_rate = np.mean(metrics['proportional_hit_rate']) if metrics['proportional_hit_rate'] else 0.0
    avg_precision = np.mean(metrics['precision']) if metrics['precision'] else 0.0
    avg_recall = np.mean(metrics['recall']) if metrics['recall'] else 0.0
    avg_ndcg = np.mean(metrics['ndcg']) if metrics['ndcg'] else 0.0
    avg_rmse = np.mean(metrics['rmse']) if metrics['rmse'] else 0.0
    avg_mae = np.mean(metrics['mae']) if metrics['mae'] else 0.0
    
    return {'avg_proportional_hit_rate': avg_proportional_hit_rate,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_ndcg': avg_ndcg,
            'avg_rmse': avg_rmse,
            'avg_mae': avg_mae}

# Step 6: t-SNE Visualization of Embeddings
def visualize_embeddings_tsne(embeddings):
    """
    Visualize high-dimensional embeddings using t-SNE.
    
    Parameters:
      - embeddings: List or numpy array of high-dimensional embeddings.
      
    Returns:
      - A scatter plot showing the reduced embeddings in a 2D space.
    """
    
    # Convert list of embeddings to a numpy array (if not already one)
    embedding_array = np.vstack(embeddings)
    
    # Apply t-SNE to reduce dimensionality to two dimensions
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embedding_array)
    
    # Create a scatter plot for the reduced embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, edgecolors='k')
    plt.title("t-SNE Visualization of Place Embeddings using DISTIL-BERT")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()  # Removed erroneous '(2D space)'

def main():
    # File paths for descriptions and ratings
    description_file = './Model/Data/PreparedData.csv'
    rating_file = './Model/Data/all_ratings.csv'
    
    print("Loading data...")
    descriptions, ratings = load_data(description_file, rating_file)
    
    print("Loading DistilBERT model...")
    tokenizer, model = load_distilbert_model()
    
    print("Generating embeddings...")
    embeddings = generate_all_embeddings(descriptions.copy(), tokenizer, model)
    
    print("Evaluating system...")
    results = evaluate_system(descriptions.copy(), ratings.copy(), embeddings)
    
    print("\nEvaluation Results:")
    print(f"Average Proportional Hit Rate: {results['avg_proportional_hit_rate']}")
    print(f"Average Precision: {results['avg_precision']}")
    print(f"Average Recall: {results['avg_recall']}")
    print(f"Average NDCG: {results['avg_ndcg']}")
    print(f"Average RMSE: {results['avg_rmse']}")
    print(f"Average MAE: {results['avg_mae']}")
    
    print("\nVisualizing embeddings with t-SNE...")
    visualize_embeddings_tsne(embeddings)

if __name__ == "__main__":
   main()


