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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        test_loss = 0.0
        all_preds, all_targets = [], []
        user_item_scores = {}

        with torch.no_grad():
            for user_ids, item_ids, ratings in tqdm(
                test_loader, desc="Evaluating"
            ):
                user_ids, item_ids, ratings = [
                    t.to(device) for t in [user_ids, item_ids, ratings]
                ]
                predictions = model(user_ids, item_ids).squeeze()
                loss = criterion(predictions, ratings)
                test_loss += loss.item()

                all_preds.append(predictions)
                all_targets.append(ratings)

                # Store user-item scores for ranking metrics
                for i, user_id in enumerate(user_ids):
                    user_id = user_id.item()
                    if user_id not in user_item_scores:
                        user_item_scores[user_id] = []
                    user_item_scores[user_id].append((item_ids[i].item(), predictions[i].item(), ratings[i].item()))

        # Convert to tensors
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Compute ranking-based metrics in one pass
        hit_rates, ndcgs, precisions, recalls = [], [], [], []

        for user_id, item_data in user_item_scores.items():
            item_data.sort(key=lambda x: x[1], reverse=True)  # Sort by predicted score
            recommended_items = [x[0] for x in item_data[:TOP_K]]
            actual_items = [x[0] for x in item_data if x[2] >= 2.5]

            if actual_items:
                hit_rate = calculate_hit_rate(recommended_items, actual_items)
                precision, recall = calculate_precision_recall(recommended_items, actual_items, TOP_K)
                ndcg = calculate_ndcg(recommended_items, actual_items, TOP_K)

                hit_rates.append(hit_rate)
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        metrics["hit_rates"].append(np.mean(hit_rates))
        metrics["ndcgs"].append(np.mean(ndcgs))
        metrics["precisions"].append(np.mean(precisions))
        metrics["recalls"].append(np.mean(recalls))
        metrics["rmses"].append(calculate_rmse(all_preds, all_targets))
        metrics["maes"].append(calculate_mae(all_preds, all_targets))

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        print(f"Hit Rate@{TOP_K}: {metrics['hit_rates'][-1]:.4f}")
        print(f"NDCG@{TOP_K}: {metrics['ndcgs'][-1]:.4f}")
        print(f"Precision@{TOP_K}: {metrics['precisions'][-1]:.4f}")
        print(f"Recall@{TOP_K}: {metrics['recalls'][-1]:.4f}")
        print(f"RMSE: {metrics['rmses'][-1]:.4f}, MAE: {metrics['maes'][-1]:.4f}")

    print(f"--------Saving Model to {MODEL_SAVE_PATH}---------")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return train_losses, test_losses, metrics


def evaluate_topn(model, test_loader, num_items, top_k, device):
    """Evaluate model using top-K metrics"""
    model.eval()
    hits, ndcgs, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_ids, item_ids, ratings = (t.to(device) for t in batch)

            # Identify positive items based on the threshold (ratings >= 4 are considered relevant)
            positive_mask = ratings >= 4.5
            positive_items = item_ids[positive_mask].cpu().numpy().flatten().tolist()

            # Skip batch if no relevant items
            if len(positive_items) == 0:
                continue  # Skip if no relevant items are found

            # Generate predictions for all user-item pairs in the batch
            all_items = torch.arange(num_items, device=device).unsqueeze(0).expand(len(user_ids), num_items)
            user_ids_batch = user_ids.unsqueeze(1).expand_as(all_items)  # Replicate user_ids for all items
            predictions = model(user_ids_batch.flatten(), all_items.flatten()).view(len(user_ids), num_items)

            # Process each user's predictions
            for idx, preds in enumerate(predictions):
                # Get top-K recommended items
                top_items = np.argsort(preds.cpu().numpy())[-top_k:][::-1]

                # Ensure positive_items is a list (relevant items for comparison)
                actual_items = positive_items

                # Calculate Hit Rate
                hit_rate = calculate_hit_rate(top_items, actual_items)
                hits.append(hit_rate)

                # Calculate NDCG
                ndcg = calculate_ndcg(top_items, actual_items, top_k)
                ndcgs.append(ndcg)

                # Calculate Precision and Recall
                precision, recall = calculate_precision_recall(top_items, actual_items, top_k)
                precisions.append(precision)
                recalls.append(recall)

    # If no hits were recorded, return zero for all metrics
    if len(hits) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Calculate final metrics
    hit_rate = np.mean(hits)
    mean_ndcg = np.mean(ndcgs)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    return hit_rate, mean_ndcg, mean_precision, mean_recall
