import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        user_input = batch['user']
        item_input = batch['item']
        target = batch['rating']
        
        optimizer.zero_grad()
        output = model(user_input, item_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            user_input = batch['user']
            item_input = batch['item']
            target = batch['rating']
            
            output = model(user_input, item_input)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
