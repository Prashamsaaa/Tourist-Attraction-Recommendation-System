import torch
import numpy as np
from tqdm import tqdm
from metrics import (
    calculate_hit_rate,
    calculate_ndcg,
    calculate_precision_recall,
    calculate_rmse,
    calculate_mae,
)
from config import *


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    num_items,
):
    train_losses = []
    test_losses = []
    metrics = {
        "hit_rates": [],
        "ndcgs": [],
        "rmses": [],
        "precisions": [],
        "recalls": [],
        "maes": [],
    }

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for user_ids, item_ids, ratings in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            user_ids, item_ids, ratings = [
                t.to(device) for t in [user_ids, item_ids, ratings]
            ]

            optimizer.zero_grad()
            predictions = model(user_ids, item_ids)

            loss = criterion(predictions.squeeze(), ratings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        test_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = [
                    t.to(device) for t in [user_ids, item_ids, ratings]
                ]
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions.squeeze(), ratings)
                test_loss += loss.item()

                all_preds.append(predictions)
                all_targets.append(ratings)

        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        hit_rate, ndcg, precision, recall = evaluate_topn(
            model, test_loader, num_items, top_k=TOP_K, device=DEVICE
        )

        metrics["hit_rates"].append(hit_rate)
        metrics["ndcgs"].append(ndcg)
        metrics["precisions"].append(precision)
        metrics["recalls"].append(recall)
        metrics["rmses"].append(calculate_rmse(all_preds, all_targets))
        metrics["maes"].append(calculate_mae(all_preds, all_targets))

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        print(f"Hit Rate@10: {hit_rate:.4f}")
        print(f"NDCG@10: {ndcg:.4f}")
        print(f"Precision@10: {precision:.4f}")
        print(f"Recall@10: {recall:.4f}")
        print(f"RMSE: {metrics['rmses'][-1]:.4f}, MAE: {metrics['maes'][-1]:.4f}")

        # scheduler.step()

    print(f"--------Saving Model to {MODEL_SAVE_PATH}---------")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return train_losses, test_losses, metrics


def evaluate_topn(model, test_loader, num_items, top_k, device):
    """Evaluate model using top-K metrics"""
    model.eval()
    hits, ndcgs, precisions, recalls = [], [], [], []

    with torch.no_grad():
        all_user_ids, all_item_ids, all_ratings = [], [], []
        
        # Collect all user, item, and rating pairs for evaluation
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_ids, item_ids, ratings = (t.to(device) for t in batch)
            all_user_ids.append(user_ids)
            all_item_ids.append(item_ids)
            all_ratings.append(ratings)

        # Concatenate all batches for predictions
        all_user_ids = torch.cat(all_user_ids)
        all_item_ids = torch.cat(all_item_ids)
        all_ratings = torch.cat(all_ratings)

        # Generate all predictions at once for the whole batch
        predictions = model(all_user_ids, all_item_ids).squeeze()

        # Evaluate each user in the batch
        for idx in range(len(all_user_ids)):
            user_id = all_user_ids[idx].item()
            item_id = all_item_ids[idx].item()
            pred = predictions[idx].item()
            rating = all_ratings[idx].item()  # Get actual rating for the item

            # Assuming relevant items are those with a rating >= 4 (adjust based on your dataset)
            positive_mask = all_ratings >= 4  # Mask for relevant items (you can adjust this threshold)
            positive_items = all_item_ids[positive_mask].cpu().numpy().flatten().tolist()

            if not positive_items:
                continue  # Skip if no relevant items are found

            # Get top-K recommended items based on predictions
            top_items = np.argsort(predictions.cpu().numpy())[-top_k:][::-1]

            # Calculate Hit Rate: Whether a relevant item is in the top-K recommended items
            hit_rate = 1 if item_id in top_items else 0
            hits.append(hit_rate)

            # Calculate NDCG: Normalize Discounted Cumulative Gain
            ndcg = calculate_ndcg(top_items, positive_items, ratings.cpu().numpy(), k = top_k)
            ndcgs.append(ndcg)

            # Calculate Precision: Proportion of relevant items in top-K
            precision = len(set(top_items) & set(positive_items)) / top_k
            precisions.append(precision)

            # Calculate Recall: Proportion of relevant items found in top-K
            recall = len(set(top_items) & set(positive_items)) / len(positive_items) if positive_items else 0
            recalls.append(recall)

    # Return average of all metrics
    hit_rate = np.mean(hits) if hits else 0
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0
    mean_precision = np.mean(precisions) if precisions else 0
    mean_recall = np.mean(recalls) if recalls else 0

    return hit_rate, mean_ndcg, mean_precision, mean_recall
