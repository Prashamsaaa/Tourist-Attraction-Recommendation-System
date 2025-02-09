# tune_hyperparams.py
import torch
import torch.optim as optim
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from optuna.visualization import plot_optimization_history, plot_param_importances
import os

# Configuration
SEED = 42
NUM_EPOCHS = 100
TEST_SIZE = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Configuration
class NCF(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dim=64, hidden_layers=[128, 64, 32], dropout=0.2):
        super().__init__()
        self.user_embed = torch.nn.Embedding(num_users, latent_dim)
        self.item_embed = torch.nn.Embedding(num_items, latent_dim)
        
        layers = []
        input_size = latent_dim * 2
        for output_size in hidden_layers:
            layers.extend([
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
            input_size = output_size
        
        layers.append(torch.nn.Linear(input_size, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        concatenated = torch.cat([user_vec, item_vec], dim=-1)
        return self.mlp(concatenated).squeeze()

class NCFDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Metrics
def calculate_rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets)**2)).item()

def calculate_mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def calculate_hit_rate(recommended, actual):
    return len(set(recommended) & set(actual)) > 0

def calculate_ndcg(recommended, actual, ratings, k=10):
    ideal = sorted(ratings, reverse=True)[:k]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(ratings[:k]))
    ideal_dcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def calculate_precision_recall(recommended, actual, k=10):
    relevant = set(actual)
    top_k = set(recommended[:k])
    precision = len(top_k & relevant) / k
    recall = len(top_k & relevant) / len(relevant) if len(relevant) > 0 else 0
    return precision, recall

# Core Functions
def load_and_preprocess_data(ratings_path, attractions_path):
    ratings = pd.read_csv(ratings_path)
    attractions = pd.read_csv(attractions_path)
    ratings = ratings.dropna(subset=["user_id", "id", "rating"])
    ratings = ratings.drop_duplicates(subset=["user_id", "id"])
    return ratings, attractions

def create_train_test_split(ratings_df, test_size):
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=SEED,
    )
    
    common_users = set(train_df["user_id"]).intersection(set(test_df["user_id"]))
    common_items = set(train_df["id"]).intersection(set(test_df["id"]))
    test_df = test_df[test_df["user_id"].isin(common_users) & test_df["id"].isin(common_items)]
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def objective(trial, train_df, test_df, num_users, num_items):
    try:
        params = {
            'latent_dim': trial.suggest_int("latent_dim", 16, 64),
            'num_layers': trial.suggest_int("num_layers", 1, 2),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            'dropout': trial.suggest_float("dropout", 0.1, 0.3),
            'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical("batch_size", [64, 128])
        }
        
        hidden_layers = [trial.suggest_int(f"hidden_layer_{i}", 32, 128) 
                        for i in range(params['num_layers'])]
        
        # DataLoaders
        train_loader = DataLoader(
            NCFDataset(
                torch.LongTensor(train_df["user_id"]),
                torch.LongTensor(train_df["id"]),
                torch.FloatTensor(train_df["rating"])
            ), 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        
        # Model
        model = NCF(num_users, num_items, 
                   latent_dim=params['latent_dim'],
                   hidden_layers=hidden_layers,
                   dropout=params['dropout']).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), 
                             lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])
        criterion = torch.nn.MSELoss()

        # Training
        for epoch in range(NUM_EPOCHS):
            model.train()
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(DEVICE), items.to(DEVICE), ratings.to(DEVICE)
                optimizer.zero_grad()
                preds = model(users, items)
                
                if torch.isnan(preds).any():
                    raise ValueError("NaN predictions detected")
                
                loss = criterion(preds, ratings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluation
        model.eval()
        all_preds, all_targets = [], []
        user_data = {}
        with torch.no_grad():
            test_loader = DataLoader(
                NCFDataset(
                    torch.LongTensor(test_df["user_id"]),
                    torch.LongTensor(test_df["id"]),
                    torch.FloatTensor(test_df["rating"])
                ),
                batch_size=params['batch_size']
            )
            
            for users, items, ratings in test_loader:
                users, items = users.to(DEVICE), items.to(DEVICE)
                preds = model(users, items).cpu()
                all_preds.append(preds)
                all_targets.append(ratings.cpu())
                
                # Collect recommendations
                for u, i, p, r in zip(users.cpu(), items.cpu(), preds, ratings.cpu()):
                    uid = u.item()
                    if uid not in user_data:
                        user_data[uid] = {'preds': [], 'actual': []}
                    user_data[uid]['preds'].append((i.item(), p.item()))
                    user_data[uid]['actual'].append((i.item(), r.item()))

        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        rmse = calculate_rmse(all_preds, all_targets)
        mae = calculate_mae(all_preds, all_targets)
        
        # Recommendation metrics
        k = 10
        hit_rate, ndcg, precision, recall = 0, 0, 0, 0
        for uid in user_data:
            pred_items = sorted(user_data[uid]['preds'], key=lambda x: x[1], reverse=True)[:k]
            actual_items = user_data[uid]['actual']
            
            hit_rate += calculate_hit_rate([i[0] for i in pred_items], 
                                         [i[0] for i in actual_items])
            ndcg += calculate_ndcg([i[0] for i in pred_items], 
                                 [i[0] for i in actual_items], 
                                 [i[1] for i in actual_items], k)
            p, r = calculate_precision_recall([i[0] for i in pred_items], 
                                            [i[0] for i in actual_items], k)
            precision += p
            recall += r
            
        num_users_eval = len(user_data)
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("hit_rate", hit_rate / num_users_eval)
        trial.set_user_attr("ndcg", ndcg / num_users_eval)
        trial.set_user_attr("precision", precision / num_users_eval)
        trial.set_user_attr("recall", recall / num_users_eval)

        return float(rmse)
    
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)[:200]}")
        return float('nan')

def visualize_results(study):
    trials_df = study.trials_dataframe()
    valid_trials = trials_df[trials_df['state'] == 'COMPLETE']
    
    # Metrics comparison
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(valid_trials['number'], valid_trials['value'], label='RMSE')
    ax[0].set_title('Validation RMSE')
    ax[0].set_xlabel('Trial')
    
    ax[1].plot(valid_trials['user_attrs_rmse'], label='RMSE')
    ax[1].plot(valid_trials['user_attrs_mae'], label='MAE')
    ax[1].set_title('Reconstruction Metrics')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Optuna visualizations
    plot_optimization_history(study).show()
    plot_param_importances(study).show()

def tune_hyperparameters(train_df, test_df, num_users, num_items):
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective(trial, train_df, test_df, num_users, num_items),
        n_trials=20,
        callbacks=[lambda study, trial: torch.cuda.empty_cache()]
    )
    return study

if __name__ == "__main__":
    # Load and preprocess
    ratings, _ = load_and_preprocess_data("Model/Data/all_ratings.csv", 
                                        "Model/Data/PreparedData.csv")
    
    # Split data
    train_df, test_df = create_train_test_split(ratings, TEST_SIZE)
    
    # Encode IDs
    user_encoder = LabelEncoder().fit(train_df["user_id"])
    item_encoder = LabelEncoder().fit(train_df["id"])
    
    train_df.loc[:, "user_id"] = user_encoder.transform(train_df["user_id"])
    test_df = test_df[test_df["user_id"].isin(user_encoder.classes_)]
    test_df.loc[:, "user_id"] = user_encoder.transform(test_df["user_id"])
    
    train_df.loc[:, "id"] = item_encoder.transform(train_df["id"])
    test_df = test_df[test_df["id"].isin(item_encoder.classes_)]
    test_df.loc[:, "id"] = item_encoder.transform(test_df["id"])
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Verify
    print(f"Training users: {num_users}, items: {num_items}")
    print(f"Test users: {test_df['user_id'].nunique()}, items: {test_df['id'].nunique()}")
    
    # Tune
    study = tune_hyperparameters(train_df, test_df, num_users, num_items)
    print("Best params:", study.best_params)
    visualize_results(study)
