import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def calculate_average_metrics(
    hit_rates, ndcgs, rmses, precisions, recalls, maes, top_k
):
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
        "hit_rate": hit_rate_avg,
        "ndcg": ndcg_avg,
        "rmse": rmse_avg,
        "mae": mae_avg,
        "precision": precision_avg,
        "recall": recall_avg,
    }


def plot_loss_curves(train_losses, test_losses, num_epochs, output_folder):
    """
    Plot training and test loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, num_epochs + 1), train_losses, label="Training Loss", color="blue"
    )
    plt.plot(range(1, num_epochs + 1), test_losses, label="Test Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss Curve")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_folder, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved at {loss_plot_path}")
    plt.show()


def plot_mae_curve(maes, num_epochs, output_folder):
    """
    Plot Mean Absolute Error (MAE) curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), maes, label="MAE", color="green")
    plt.title("Mean Absolute Error (MAE) Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    mae_plot_path = os.path.join(output_folder, "mae_curve.png")
    plt.savefig(mae_plot_path)
    print(f"MAE curve saved at {mae_plot_path}")
    plt.show()


def plot_hit_rate_ndcg(hit_rates, ndcgs, num_epochs, top_k, output_folder):
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
    hit_ndcg_plot_path = os.path.join(output_folder, "hit_rate_ndcg_curve.png")
    plt.savefig(hit_ndcg_plot_path)
    print(f"Hit Rate & NDCG curve saved at {hit_ndcg_plot_path}")
    plt.show()
