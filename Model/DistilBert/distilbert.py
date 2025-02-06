import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertModel
import torch

class DistilBERTRecommender:
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize the DistilBERT Recommender.
        
        :param model_name: Name of the pre-trained DistilBERT model to load.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)

    def generate_embedding(self, text):
        """
        Generate an embedding for a single text using DistilBERT.
        
        :param text: The input text for which to generate an embedding.
        :return: Numpy array representing the embedding.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding for text '{text}': {e}")

    def generate_all_embeddings(self, data):
        """
        Generate embeddings for all descriptions in the dataset.
        
        :param data: DataFrame containing a 'Description' column.
        :return: List of numpy arrays representing embeddings for all descriptions.
        """
        try:
            embeddings = []
            for description in data['Description']:
                embeddings.append(self.generate_embedding(description))
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings for dataset: {e}")

    def recommend_places(self, user_id, data, ratings, embeddings, top_n=5):
            """
            Recommend places for a user based on their past ratings and content similarity.

            :param user_id: ID of the user for whom recommendations are generated.
            :param data: DataFrame containing place information (must include 'ID', 'Name', etc.).
            :param ratings: DataFrame containing user-item interaction data (must include 'user_id', 'id', and 'rating').
            :param embeddings: List of precomputed embeddings for all places.
            :param top_n: Number of recommendations to return (default=5).
            :return: DataFrame containing top-N recommended places with scores.
            """
            # print("distilbert recommendor envoked")
            # try:
            user_ratings = ratings[ratings['user_id'] == user_id]
            if user_ratings.empty:
                print(f"No ratings found for user {user_id}.")
                return pd.DataFrame()
            # Build an embedding subset for only the rows in 'data'
            # Assumes 'ID' is 1-based; adjust if needed
            valid_ids = data['ID'].astype(int).unique()
            embedding_map = {}
            for idx in valid_ids:
                # Skip IDs out of range
                if idx - 1 < 0 or idx - 1 >= len(embeddings):
                    continue
                embedding_map[idx] = embeddings[idx - 1]
            print("embedding map",embedding_map)
            recommendations = []
            
            # Iterate over rated places
            for _, row in user_ratings.iterrows():
                place_id = row['id']
                if place_id not in embedding_map:
                    continue

                place_embedding = embedding_map[place_id].reshape(1, -1)

                # Create a matrix of embeddings to match 'data' length
                # Filter out rows whose ID is missing in embedding_map
                filtered_data = data[data['ID'].isin(embedding_map.keys())].copy()
                embedding_subset = [embedding_map[item_id] for item_id in filtered_data['ID']]
                embedding_subset = np.vstack(embedding_subset)

                similarities = cosine_similarity(place_embedding, embedding_subset).flatten()
                filtered_data['Similarity_Score'] = similarities

                # Weighted combination of similarity and rating
                filtered_data['Recommendation_Score'] = 0.7 * filtered_data['Similarity_Score'] + 0.3 * row['rating']

                recommended_places = filtered_data.nlargest(top_n, 'Recommendation_Score').copy()
                recommended_places.rename(columns={'Recommendation_Score': 'DistilBERT_Score'}, inplace=True)
                recommendations.append(recommended_places)
            
            final_recommendations = pd.concat(recommendations).drop_duplicates(subset=['ID']).nlargest(top_n, 'DistilBERT_Score')
            return final_recommendations[['ID', 'Name', 'Description', 'Province', 'Tags', 'DistilBERT_Score']]
            
        # except Exception as e:
        #     raise RuntimeError(f"Error generating recommendations for user {user_id}: {e}")
