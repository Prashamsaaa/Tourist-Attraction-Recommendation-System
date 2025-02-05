import numpy as np
import matplotlib.pyplot as plt

def calculate_average_metrics(hit_rates, ndcgs, rmses, precisions, recalls, maes, top_k):
    """
    Calculate average metrics over all epochs.
    """
    hit_rate_avg = np.mean(hit_rates)
    ndcg_avg = np.mean(ndcgs)
    rmse_avg = np.mean(rmses)
    precision_avg = np.mean(precisions)
    recall_avg = np.mean(recalls)
    mae_avg = np.mean(maes)

    print(f"Average Hit Rate@{top_k}: {hit_rate_avg:.4f}")
    print(f"Average NDCG@{top_k}: {ndcg_avg:.4f}")
    print(f"Average RMSE: {rmse_avg:.4f}")
    print(f"Average MAE: {mae_avg:.4f}")
    print(f"Average Precision@{top_k}: {precision_avg:.4f}")
    print(f"Average Recall@{top_k}: {recall_avg:.4f}")

    return {
        'hit_rate': hit_rate_avg,
        'ndcg': ndcg_avg,
        'rmse': rmse_avg,
        'mae': mae_avg,
        'precision': precision_avg,
        'recall': recall_avg
    }

def plot_loss_curves(train_losses, test_losses, num_epochs):
    """
    Plot training and test loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label="Training Loss", color="blue")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss", color="red")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_mae_curve(maes, num_epochs):
    """
    Plot Mean Absolute Error (MAE) curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), maes, label='MAE', color='green')
    plt.title('Mean Absolute Error (MAE) Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hit_rate_ndcg(hit_rates, ndcgs, num_epochs, top_k):
    """
    Plot Hit Rate and NDCG curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), hit_rates, label=f"Hit Rate@{top_k}", color="green")
    plt.plot(range(1, num_epochs + 1), ndcgs, label=f"NDCG@{top_k}", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Hit Rate and NDCG")
    plt.legend()
    plt.grid(True)
    plt.show()
