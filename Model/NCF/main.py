from config import *
from models import NCF
from dataset import load_and_preprocess_data, create_train_test_split, create_dataset_and_loaders
from train_eval import train_model, evaluate_topn
from utils import plot_loss_curves, calculate_average_metrics
from dynamic_model_manager import DynamicModelManager
from recommendation import generate_recommendations

def main():
    # Load and preprocess data
    ratings_df, attraction_df, user_encoder, place_encoder = load_and_preprocess_data(
        './Data/FinalDataset/ratings.csv', 
        './Data/FinalDataset/Data.csv'
    )
    num_users = len(user_encoder.classes_)
    num_items = len(place_encoder.classes_)
    # Split data
    train_df, test_df = create_train_test_split(ratings_df, test_size=TEST_SIZE)
    train_loader, test_loader = create_dataset_and_loaders(train_df, test_df, BATCH_SIZE)
    
    print("Initializing the NCF model...........")
    # Initialize model
    model = NCF(
        num_users=len(user_encoder.classes_),
        num_items=len(place_encoder.classes_),
        latent_dim=LATENT_DIM
    ).to(DEVICE)
    print(f"Model: {model}")
    
    # Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1, betas=(0.9, 0.999))
    train_losses, test_losses, metrics = train_model(
        model, train_loader, test_loader, 
        criterion, optimizer, NUM_EPOCHS, DEVICE, num_items
    )
    print("-------------------Training Completed-----------------------")
    print("Printing Average Metrics and Displaying Loss Curves")
    calculate_average_metrics(metrics['hit_rates'],metrics['ndcgs'], metrics['rmses'], metrics['precisions'], metrics['recalls'], metrics['maes'], TOP_K)
    plot_loss_curves(train_losses=train_losses, test_losses=test_losses, num_epochs= NUM_EPOCHS)
    print("Initializing Model Manager")
    # Initialize dynamic manager
    model_manager = DynamicModelManager(model, optimizer, criterion, user_encoder, place_encoder)
    
    # Generate recommendations
    original_user_id = 150
    encoded_user_id = user_encoder.transform([original_user_id])[0]
    recommendations = generate_recommendations(
        model_manager.model, encoded_user_id, 
        num_items, TOP_K, DEVICE,
        place_encoder, attraction_df
    )
    
    # Print recommendations
    print(f"\nTop {TOP_K} recommendations for user {original_user_id}:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['name']} (Confidence: {rec['confidence']:.3f})")

if __name__ == "__main__":
    main()
