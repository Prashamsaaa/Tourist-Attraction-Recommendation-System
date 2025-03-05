import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import necessary modules
from ContentBased.content_based import ContentBasedRecommender
from DistilBert.distilbert import DistilBERTRecommender
from NCF.recommendation import generate_recommendations
from Hybrid.hybrid_recommender import HybridRecommender
from NCF.dataset import load_and_preprocess_data, create_dataset_and_loaders, NCFDataset, create_train_test_split
from NCF.models import NCF, GMF
from NCF.config import *
from NCF.train_eval import train_model
from NCF.dynamic_model_manager import DynamicModelManager
from NCF.utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data paths configuration
DATA_PATHS = {
    'content': "Model/Data/PreparedData.csv",
    'ratings': "Model/Data/all_ratings.csv",
    'prepared_data': './Model/Data/PreparedData.csv',
    'tags': './Model/Data/CategorizedTags.json',
    'embeddings': './Model/models/distilbert_embeddings.npy',
    'model': './Model/models/model.pth'
}

def validate_input(user_input, valid_options):
    """Validate user input against a list of valid options."""
    if user_input not in valid_options:
        raise ValueError(f"Invalid input: '{user_input}'. Please choose from {valid_options}.")

def get_category_tags(content_recommender, category):
    """Retrieve tags for a specific category."""
    if category in content_recommender.categorized_tags:
        return sorted(content_recommender.categorized_tags[category])
    return []

def load_data_and_models():
    """Load all necessary data and models for the recommendation system."""
    # Validate data file existence
    if not all(os.path.exists(DATA_PATHS[p]) for p in ['prepared_data', 'tags']):
        raise FileNotFoundError("Required data files are missing.")

    # Content-Based Recommender
    content_recommender = ContentBasedRecommender(DATA_PATHS['prepared_data'], DATA_PATHS['tags'])

    # DistilBERT Recommender
    distilbert_recommender = DistilBERTRecommender()
    descriptions = pd.read_csv(DATA_PATHS['content'])
    
    # Load or generate embeddings
    if os.path.exists(DATA_PATHS['embeddings']):
        embeddings = np.load(DATA_PATHS['embeddings'], allow_pickle=True)
    else:
        embeddings = distilbert_recommender.generate_all_embeddings(descriptions)
    embeddings = embeddings[:len(descriptions)] # Truncate embeddings to match DataFrame length
    descriptions['embeddings'] = embeddings.tolist()

    # NCF Recommender and Data Preparation
    ratings_df, user_encoder, place_encoder = load_and_preprocess_data(
        DATA_PATHS['ratings']
    )

    # Create train-test split and data loaders
    train_df, test_df = create_train_test_split(ratings_df, test_size=TEST_SIZE)
    train_loader, test_loader = create_dataset_and_loaders(train_df, test_df, BATCH_SIZE)

    # Initialize NCF Model
    ncf_recommender = NCF(
        num_users=len(user_encoder.classes_), 
        num_items=len(place_encoder.classes_), 
        latent_dim=LATENT_DIM
    ).to(DEVICE)  # Move model to the correct device

    # Load model state from backup if available, otherwise from initial model path
    backup_path = 'model_backup.pt'
    model_path = DATA_PATHS['model']
    load_path = backup_path if os.path.exists(backup_path) else model_path
    print(f"Loading NCF model from: {load_path}") # Indicate load path
    import torch.nn as nn
    from NCF.models import MLP
    torch.serialization.add_safe_globals([NCF, GMF, nn.Embedding, MLP])
    checkpoint = torch.load(load_path, map_location=DEVICE, weights_only=False)
    ncf_recommender.load_state_dict(checkpoint , strict=False)
    print(ncf_recommender)

    return (content_recommender, distilbert_recommender, ncf_recommender,
            descriptions, ratings_df, user_encoder, place_encoder,
            train_loader, test_loader, train_df, test_df)

def setup_dynamic_model_manager(ncf_recommender, ratings_df, user_encoder, place_encoder):
    """Setup dynamic model manager for continuous learning."""
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ncf_recommender.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Initialize Dynamic Model Manager
    dynamic_model_manager = DynamicModelManager(
        model=ncf_recommender,
        optimizer=optimizer,
        criterion=criterion,
        user_encoder=user_encoder,
        place_encoder=place_encoder
    )

    return dynamic_model_manager

def main():
    """Main function to run the dynamic recommendation system."""
    try:
        # Set random seed for reproducibility
        set_seed(SEED)

        # Load data and models
        (content_recommender, distilbert_recommender, ncf_recommender, 
         descriptions, ratings, user_encoder, place_encoder, 
         train_loader, test_loader, train_df, test_df) = load_data_and_models()
        
        # Setup Hybrid Recommender
        hybrid_recommender = HybridRecommender(
            content_recommender, 
            distilbert_recommender, 
            ncf_recommender
        )

        # Setup Dynamic Model Manager
        dynamic_model_manager = setup_dynamic_model_manager(
            ncf_recommender, ratings, user_encoder, place_encoder
        )

        # Get available provinces and categories
        provinces = content_recommender.data['Province'].unique()
        categories = content_recommender.data['Category'].unique()

        while True:
            print("\n--- Dynamic Recommendation System ---")
            print("1. Recommendation")
            print("2. Update Model")
            print("3. View Model Stats")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()

            if choice == "1":
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
                            descriptions_all=descriptions,
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

            elif choice == "2":
                # Dynamic Model Update
                print("\n--- Update Model ---")
                try:
                    print("Enter a new rating to update the model")
                    user_id = int(input("User ID: "))
                    item_id = int(input("Item ID: "))
                    rating = float(input("Rating (0-5): "))

                    # Validate rating
                    if not (0 <= rating <= 5):
                        raise ValueError("Rating must be between 0 and 5")

                    # Prepare rating data
                    new_rating_data = {
                        'user_id': user_id,
                        'id': item_id,
                        'rating': rating
                    }

                    # Update model
                    update_success = dynamic_model_manager.update_model(
                        new_rating_data, 
                        train_dataset=train_loader.dataset, 
                        device=DEVICE
                    )

                    if update_success:
                        print("Model updated successfully!")
                        # Optional: Retrain or validate the model incrementally
                        dynamic_model_manager.save_model_state()
                    else:
                        print("Model update failed.")

                except ValueError as e:
                    print(f"Input Error: {e}")

            elif choice == "3":
                # View Model Statistics
                model_stats = dynamic_model_manager.get_model_stats()
                print("\n--- Model Statistics ---")
                for key, value in model_stats.items():
                    print(f"{key}: {value}")

            elif choice == "4":
                # Exit the system
                print("Exiting the Dynamic Recommendation System. Goodbye!")
                break

            else:
                print("Invalid choice. Please enter a number between 1 and 4.")

    except Exception as e:
        logger.error(f"System Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()