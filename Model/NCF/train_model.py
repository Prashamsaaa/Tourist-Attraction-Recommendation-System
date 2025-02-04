# train_model.py

import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import logging

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=30, top_k=5, device=None):
    if device:
        model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user_ids, item_ids, ratings in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if device:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(user_ids, item_ids).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
