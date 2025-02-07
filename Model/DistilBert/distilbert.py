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
            print(user_ratings)
            # Assumes 'ID' is 1-based; adjust if needed
            valid_ids = data['id'].astype(int).unique()
            print(f"valid ids: {len(valid_ids)}")
            embedding_map = {}
            # embeddings = np.array(embeddings)
            for i, idx in enumerate(valid_ids):
                # Skip IDs out of range
                if i >= len(embeddings):
                    print(f"ID out of range {i}")
                    continue
                embedding_map[idx] = embeddings[idx - 1]
            print("embedding map",len(embedding_map))

            # First convert all embeddings in the map to numpy arrays
            for idx in embedding_map:
                embedding_map[idx] = np.array(embedding_map[idx])

            # # Verify the conversion
            # first_key = list(embedding_map.keys())[0]
            # print(f"Type of embedding: {type(embedding_map[first_key])}")
            # print(f"Shape of embedding: {embedding_map[first_key].shape}")

            recommendations = []
            
            # Iterate over rated places
            for _, row in user_ratings.iterrows():
                place_id = row['id']
                print(f"place_id: {place_id}")
                if place_id not in embedding_map:
                    print('place_id not in embedding_map')
                    return "User has not rated place for given preference"
                    continue
                
                place_embedding = embedding_map[place_id].reshape(1, -1)

                # Create a matrix of embeddings to match 'data' length
                # # Filter out rows whose ID is missing in embedding_map
                filtered_data = data[data['id'].isin(embedding_map.keys())].copy()
                # filtered_data = data
                embedding_subset = [embedding_map[item_id] for item_id in filtered_data['id']]
                embedding_subset = np.vstack(embedding_subset)
                similarities = cosine_similarity(place_embedding, embedding_subset).flatten()
                filtered_data['Similarity_Score'] = similarities
                print("UNDERSTAND THIS")
                # Weighted combination of similarity and rating
                filtered_data['Recommendation_Score'] = 0.7 * filtered_data['Similarity_Score'] + 0.3 * row['rating']

                recommended_places = filtered_data.nlargest(top_n, 'Recommendation_Score').copy()
                recommended_places.rename(columns={'Recommendation_Score': 'DistilBERT_Score'}, inplace=True)
                recommendations.append(recommended_places)
            final_recommendations = pd.concat(recommendations).drop_duplicates(subset=['id']).nlargest(top_n, 'DistilBERT_Score')
            print(final_recommendations)
            return final_recommendations[['id', 'Name', 'Description', 'Province', 'Tags', 'DistilBERT_Score']]
            
        # except Exception as e:
        #     raise RuntimeError(f"Error generating recommendations for user {user_id}: {e}")
