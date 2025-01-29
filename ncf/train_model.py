import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np

def evaluate_topn(model, test_loader, num_items, top_k):
    """
    Evaluate model performance using Hit Rate, NDCG@K, RMSE, Precision, and Recall.

    Args:
        model: The trained recommendation model
        test_loader: DataLoader containing test data
        num_items: Total number of items in the dataset
        top_k: Number of top items to consider for evaluation

    Returns:
        tuple: (hit_rate, ndcg, rmse, precision, recall) metrics
    """
    device = next(model.parameters()).device
    hits = []
    ndcgs = []
    squared_errors = []
    precisions = []
    recalls = []

    top_k = min(top_k, num_items)
    model.eval()

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            # Threshold for relevant items (adjust as needed)
            relevant_threshold = ratings.mean().item()

            # Get positive items (actual interactions)
            positive_items = item_ids[ratings >= relevant_threshold].tolist()

            if not positive_items:
                continue

            # Generate predictions for all items for each user
            all_item_ids = torch.arange(num_items).to(device)

            # Handle each user in the batch separately
            for user_id in user_ids:
                user_vector = user_id.repeat(num_items)
                predictions = model(user_vector, all_item_ids)
                predictions = predictions.squeeze()

                try:
                    # Get top-k item predictions
                    _, indices = torch.topk(predictions, k=top_k)
                    recommended_items = indices.cpu().tolist()

                    # Calculate Hit Rate
                    hit = int(any(item in recommended_items for item in positive_items))
                    hits.append(hit)

                    # Calculate NDCG
                    dcg = 0.0
                    idcg = 1.0  # Ideal DCG for binary case with one positive item
                    for idx, item in enumerate(recommended_items):
                        if item in positive_items:
                            dcg += 1.0 / np.log2(idx + 2)
                    ndcg = dcg / idcg
                    ndcgs.append(ndcg)

                    # Calculate Precision and Recall
                    hit_items = set(recommended_items) & set(positive_items)
                    precision = len(hit_items) / top_k if top_k > 0 else 0
                    recall = len(hit_items) / len(positive_items) if positive_items else 0
                    precisions.append(precision)
                    recalls.append(recall)

                    # Calculate squared errors for RMSE
                    batch_predictions = model(user_ids, item_ids).squeeze()
                    batch_se = (batch_predictions - ratings) ** 2
                    squared_errors.extend(batch_se.cpu().tolist())

                except RuntimeError as e:
                    print(f"Warning: Error processing predictions: {e}")
                    continue

    if not hits:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hit_rate = np.mean(hits)
    mean_ndcg = np.mean(ndcgs)
    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    return hit_rate, mean_ndcg, rmse, mean_precision, mean_recall





def train_model(
    model, train_loader, test_loader, num_epochs, criterion, optimizer, device, top_k, num_items
):
    train_losses = []
    test_losses = []
    hit_rates, ndcgs, rmses = [], [], []
    best_metrics = {'epoch': -1, 'hit_rate': 0.0, 'ndcg': 0.0, 'rmse': float('inf')}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for user_ids, item_ids, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions.squeeze(), ratings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions.squeeze(), ratings)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Calculate evaluation metrics
        hit_rate, ndcg, rmse = evaluate_topn(model, test_loader, num_items, top_k)
        hit_rates.append(hit_rate)
        ndcgs.append(ndcg)
        rmses.append(rmse)

        if hit_rate > best_metrics['hit_rate']:
            best_metrics.update({'epoch': epoch, 'hit_rate': hit_rate, 'ndcg': ndcg, 'rmse': rmse})

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Hit Rate: {hit_rate:.4f}")

    save_loss_curve(train_losses, test_losses, hit_rates, ndcgs)
    torch.save(model.state_dict(), "trained_model.pth")
    print(f"Best metrics: {best_metrics}")

def save_loss_curve(train_losses, test_losses, hit_rates, ndcgs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(hit_rates) + 1), hit_rates, label="Hit Rate")
    plt.plot(range(1, len(ndcgs) + 1), ndcgs, label="NDCG")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics_curve.png")
    plt.close()
