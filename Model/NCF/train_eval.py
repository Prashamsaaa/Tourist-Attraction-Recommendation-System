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
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_ids, item_ids, ratings = (t.to(device) for t in batch)

            # Identify positive items based on the threshold
            positive_mask = ratings >= 4
            positive_items = item_ids[positive_mask].cpu().numpy().flatten().tolist()

            # Ensure positive_items is not empty before proceeding
            if len(positive_items) == 0:
                continue  # Skip if no relevant items are found

            # Generate predictions for all user-item pairs in the batch
            all_items = torch.arange(num_items, device=device).unsqueeze(0).expand(len(user_ids), num_items)
            user_ids_batch = user_ids.unsqueeze(1).expand_as(all_items)  # Replicate user_ids for all items
            predictions = model(user_ids_batch.flatten(), all_items.flatten()).view(len(user_ids), num_items)

            # Process each user's predictions
            for idx, preds in enumerate(predictions):
                top_items = np.argsort(preds.cpu().numpy())[-top_k:][::-1]
                actual_items = positive_items

                # Handle the case where there is only one relevant item
                if isinstance(actual_items, (np.ndarray, list)):
                    actual_items = list(actual_items)  # Ensure it's a list
                else:
                    actual_items = [int(actual_items)]  # Convert single item to list

                # Calculate Hit Rate
                hit_rate = calculate_hit_rate(top_items, actual_items)
                hits.append(hit_rate)

                # Calculate NDCG
                relevant_ratings = [ratings[i].cpu().item() for i in range(len(item_ids)) if item_ids[i] in actual_items]
                ndcg = calculate_ndcg(top_items, actual_items, relevant_ratings, top_k)
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


