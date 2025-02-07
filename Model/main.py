import os
import logging
import pandas as pd
import numpy as np
import torch
from ContentBased.content_based import ContentBasedRecommender
from DistilBert.distilbert import DistilBERTRecommender
from NCF.recommendation import generate_recommendations
from Hybrid.hybrid_recommender import HybridRecommender
from NCF.dataset import load_and_preprocess_data
from NCF.models import NCF
from NCF.config import *

logging.basicConfig(level=logging.INFO)

DATA_PATHS = {
    'content': "Model/Data/PreparedData.csv",
    'ratings': "Model/Data/all_ratings.csv",
    'prepared_data': './Model/Data/PreparedData.csv',
    'tags': './Model/Data/CategorizedTags.json',
    'embeddings': './Model/models/distilbert_embeddings.npy',
    'model': './Model/models/model.pth'
}

def validate_input(user_input, valid_options):
    if user_input not in valid_options:
        raise ValueError(f"Invalid input: '{user_input}'. Please choose from {valid_options}.")

def get_category_tags(content_recommender, category):
    if category in content_recommender.categorized_tags:
        return sorted(content_recommender.categorized_tags[category])
    return []

def load_data_and_models():
    # Content-Based Recommender
    if not all(os.path.exists(DATA_PATHS[p]) for p in ['prepared_data', 'tags']):
        raise FileNotFoundError("Required data files are missing.")
    content_recommender = ContentBasedRecommender(DATA_PATHS['prepared_data'], DATA_PATHS['tags'])

    # DistilBERT Recommender
    distilbert_recommender = DistilBERTRecommender()
    descriptions = pd.read_csv(DATA_PATHS['content'])
    # Load or generate embeddings
    if os.path.exists(DATA_PATHS['embeddings']):
        embeddings = np.load(DATA_PATHS['embeddings'], allow_pickle=True)
    else:
        embeddings = distilbert_recommender.generate_all_embeddings(descriptions)
    # np.save(DATA_PATHS['embeddings'], embeddings)
    descriptions['embeddings'] = embeddings.tolist()
    # print(descriptions['embeddings'])
    # descriptions.to_csv(DATA_PATHS['content'])
    # print("Embeddings saved")

    # NCF Recommender
    ratings_df, attraction_df, user_encoder, place_encoder = load_and_preprocess_data(
        DATA_PATHS['ratings'], 
        DATA_PATHS['content']
    )
    
    ncf_recommender = NCF(
        num_users=len(user_encoder.classes_), 
        num_items=len(place_encoder.classes_), 
        latent_dim=LATENT_DIM
    )
    
    checkpoint = torch.load(DATA_PATHS['model'], map_location='cpu')
    ncf_recommender.load_state_dict(checkpoint if isinstance(checkpoint, dict) else checkpoint)

    return (content_recommender, distilbert_recommender, ncf_recommender, 
            descriptions, ratings_df, user_encoder, place_encoder)

def main():
    try:
        (content_recommender, distilbert_recommender, ncf_recommender, 
         descriptions, ratings, user_encoder, place_encoder) = load_data_and_models()
        
        hybrid_recommender = HybridRecommender(
            content_recommender, 
            distilbert_recommender, 
            ncf_recommender
        )

        provinces = content_recommender.data['Province'].unique()
        categories = content_recommender.data['Category'].unique()

        while True:
            print("\n--- Recommendation System ---")
            user_type = input("Are you a new user or an old user? (new/old): ").strip().lower()

            if user_type == "new":
                try:
                    print(f"Available Provinces: {provinces}")
                    province = input("Enter your preferred province: ").strip()
                    validate_input(province, provinces)

                    print(f"\nAvailable Categories: {categories}")
                    category = input("Enter your preferred category: ").strip()
                    validate_input(category, categories)

                    category_tags = get_category_tags(content_recommender, category)
                    if not category_tags:
                        print(f"No tags found for category '{category}'")
                        continue

                    print(f"\nAvailable Tags for {category}:")
                    for i in range(0, len(category_tags), 5):
                        print(", ".join(category_tags[i:i+5]))

                    tags_input = input("\nEnter your preferred tags (comma-separated): ").strip()
                    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

                    invalid_tags = [tag for tag in tags if tag not in category_tags]
                    if invalid_tags:
                        raise ValueError(f"Invalid tags: {invalid_tags}")

                    recommendations = hybrid_recommender.recommend_for_new_user(
                        province=province,
                        category=category,
                        tags=tags
                    )

                    if recommendations.empty:
                        print("\nNo recommendations found for your preferences.")
                    else:
                        print("\n--- Recommendations for New User ---")
                        print(recommendations.to_string(index=False))

                except ValueError as e:
                    print(f"Error: {e}")

            elif user_type == "old":
                try:
                    user_id = int(input("Enter your User ID: "))
                    if user_id < 0:
                        raise ValueError("User ID must be a positive number")

                    print(f"\nAvailable Provinces: {provinces}")
                    preferred_province = input("Enter your preferred province: ").strip()
                    validate_input(preferred_province, provinces)

                    recommendations = hybrid_recommender.recommend_for_old_user(
                        user_id=user_id,
                        descriptions=descriptions,
                        ratings=ratings,
                        preferred_province=preferred_province,
                        user_encoder=user_encoder,
                        place_encoder=place_encoder
                    )

                    if recommendations.empty:
                        print("\nNo recommendations found for your preferences.")
                    else:
                        print("\n--- Recommendations for Old User ---")
                        print(recommendations.to_string(index=False))

                except ValueError as e:
                    print(f"Input Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
            else:
                print("Invalid user type. Please enter 'new' or 'old'.")

    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    main()
