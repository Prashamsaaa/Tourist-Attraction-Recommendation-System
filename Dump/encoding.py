import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class EncodingModel:
    def __init__(self, data):
        # Load the encoded data
        self.data = data
        # Extract the encoded columns for provinces and categories
        self.province_encoded = data[['province_Bagmati Province', 'province_Gandaki Province', 'province_Karnali Province', 
                                      'province_Koshi Province', 'province_Lumbini Province', 'province_Madhesh Province', 
                                      'province_Sudurpashchim Province']].to_numpy()
        self.category_encoded = data[['Cultural', 'Miscellaneous', 'Historical', 'Religious', 'Recreational', 'Adventure', 'Nature']].to_numpy()
        # Combine the province and category encodings for items
        self.item_features = np.hstack([self.province_encoded, self.category_encoded])

    def encode_user_preferences(self, preferred_categories, preferred_province):
        # Encode user preferences based on the selected categories and province
        province_mask = np.zeros_like(self.province_encoded[0])  # Create a zero vector with the same length
        category_mask = np.zeros_like(self.category_encoded[0])

        # Assign 1s to the preferred provinces in the mask
        province_columns = ['province_Bagmati Province', 'province_Gandaki Province', 'province_Karnali Province', 
                            'province_Koshi Province', 'province_Lumbini Province', 'province_Madhesh Province', 
                            'province_Sudurpashchim Province']
        if f'province_{preferred_province}' in province_columns:
            province_mask[province_columns.index(f'province_{preferred_province}')] = 1

        # Assign 1s to the preferred categories in the mask
        for category in preferred_categories:
            if category in self.data.columns:
                category_mask[self.data.columns.get_loc(category) - self.data.columns.get_loc('Cultural')] = 1

        # Combine the user preferences into a single vector
        user_preferences = np.concatenate([province_mask, category_mask]).reshape(1, -1)
        return user_preferences

    def recommend_for_new_user(self, preferred_categories, preferred_province, top_n=5):
        # Get user preferences encoded as a vector
        user_preferences = self.encode_user_preferences(preferred_categories, preferred_province)

        # Compute cosine similarity between user preferences and item features
        similarities = cosine_similarity(user_preferences, self.item_features).flatten()

        # Add similarities to the data and get top recommendations
        self.data['similarity'] = similarities
        recommendations = self.data.sort_values(by='similarity', ascending=False).head(top_n)
        return recommendations[['name', 'tags', 'similarity']], similarities

    def plot_similarity_distribution(self, all_similarities):
        # Plot the similarity distribution
        plt.figure(figsize=(10, 6))
        plt.hist(all_similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# Example usage for a new user
if __name__ == "__main__":
    # Load the encoded data (adjust the path as needed)
    data = pd.read_csv('Notebook/Output/encoded_data.csv')  # Replace with actual path if necessary
    
    # Initialize the encoding model with the data
    encoding_model = EncodingModel(data)

    # User inputs
    user_preferred_categories = ["Religious", "Cultural"]
    user_preferred_province = "Karnali Province"

    # Get recommendations for the new user
    new_user_recommendations, all_similarities = encoding_model.recommend_for_new_user(
        preferred_categories=user_preferred_categories,
        preferred_province=user_preferred_province,
        top_n=5
    )

    # Display the top recommendations
    print("Top Recommendations for New User:")
    print(new_user_recommendations)

    # Plot the similarity distribution
    encoding_model.plot_similarity_distribution(all_similarities)
