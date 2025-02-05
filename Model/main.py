import os
import logging
import pandas as pd
import numpy as np
from ContentBased.content_based import ContentBasedRecommender
from DistilBert.distilbert import DistilBERTRecommender
# from NCF.config import validate_config
# from NCF.preprocess import encode_features
from NCF.recommendation import generate_recommendations
from Hybrid.hybrid_recommender import HybridRecommender

logging.basicConfig(level=logging.INFO)

def validate_input(user_input, valid_options):
    """Validate user input against valid options."""
    if user_input not in valid_options:
        raise ValueError(f"Invalid input: '{user_input}'. Please choose from {valid_options}.")

def get_category_tags(content_recommender, category):
    """Get tags that belong to the selected category."""
    if category in content_recommender.categorized_tags:
        return sorted(content_recommender.categorized_tags[category])
    return []

def main():
    # === Step 1: Initialize Content-Based Recommender === #
    print("Initializing Content-Based Recommender...")
    content_file_path = './data/PreparedData.csv'
    categorized_tags_path = './data/CategorizedTags.json'

    if not os.path.exists(content_file_path) or not os.path.exists(categorized_tags_path):
        print("Error: Required data files are missing.")
        return

    content_recommender = ContentBasedRecommender(content_file_path, categorized_tags_path)
    print("Content-Based Recommender initialized successfully.\n")

    # === Step 2: Initialize DistilBERT Recommender and Load Embeddings === #
    print("Initializing DistilBERT Recommender...")
    distilbert_recommender = DistilBERTRecommender()
    descriptions_file = './data/PreparedData.csv'
    embeddings_file = './models/distilbert_embeddings.npy'

    if not os.path.exists(descriptions_file):
        print("Error: Descriptions file is missing.")
        return

    descriptions = pd.read_csv(descriptions_file)

    # Load pre-computed embeddings if they exist, otherwise generate them
    if os.path.exists(embeddings_file):
        print("Loading pre-computed DistilBERT embeddings...")
        embeddings = np.load(embeddings_file, allow_pickle=True)
        print("DistilBERT embeddings loaded successfully.\n")
    else:
        print("Generating new DistilBERT embeddings...")
        embeddings = distilbert_recommender.generate_all_embeddings(descriptions)
        os.makedirs('./models', exist_ok=True)
        np.save(embeddings_file, embeddings)
        print("DistilBERT embeddings generated and saved successfully.\n")

    # === Step 3: Initialize NCF Recommender === #
    print("Initializing Neural Collaborative Filtering (NCF) Model...")
    
    # Load data and mappings
    df, user_mapping, item_mapping = encode_features(validate_config.DATA_PATH)

    # Load pre-trained NCF model
    ncf_model_path = './models/ncf_model.pth'
    if not os.path.exists(ncf_model_path):
        print("Error: Pre-trained NCF model not found.")
        return

    ncf_recommender = generate_recommendations(ncf_model_path, item_mapping)
    print("NCF Model loaded successfully.\n")

    # === Step 4: Initialize Hybrid Recommender === #
    print("Initializing Hybrid Recommender...")
    hybrid_recommender = HybridRecommender(content_recommender, distilbert_recommender, ncf_recommender)
    print("Hybrid Recommender initialized successfully.\n")

    # === Step 5: Simulate User Interaction === #
    
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
                    return

                print(f"\nAvailable Tags for {category}:")
                for i in range(0, len(category_tags), 5):
                    print(", ".join(category_tags[i:i+5]))

                tags_input = input("\nEnter your preferred tags (comma-separated): ").strip()
                tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

                invalid_tags = [tag for tag in tags if tag not in category_tags]
                if invalid_tags:
                    raise ValueError(f"Invalid tags: {invalid_tags}. Please choose from the available tags.")

                recommendations = hybrid_recommender.recommend_for_new_user(
                    province=province,
                    category=category,
                    tags=tags,
                )

                print("\n--- Recommendations for New User ---")
                if recommendations.empty:
                    print("No recommendations found for your preferences.")
                else:
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

                ratings_file = './data/all_ratings.csv'
                if not os.path.exists(ratings_file):
                    print("Error: Ratings file is missing.")
                    continue

                ratings = pd.read_csv(ratings_file)

                print("\nGenerating recommendations...")
                recommendations = hybrid_recommender.recommend_for_old_user(
                    user_id=user_id,
                    embeddings=embeddings,
                    descriptions=descriptions,
                    ratings=ratings,
                    preferred_province=preferred_province,
                )

                print("\n--- Recommendations for Old User ---")
                if recommendations.empty:
                    print("No recommendations found for your preferences.")
                else:
                    print("\nHere are your personalized recommendations:")
                    print(recommendations.to_string(index=False))

            except ValueError as e:
                print(f"Input Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
            else:
                print("Invalid user type. Please enter 'new' or 'old'.")
