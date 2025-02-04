import torch

def train_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.

    :param model: PyTorch model to train.
    :param train_loader: DataLoader containing training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer for updating model weights.
    :return: Average training loss for the epoch.
    """
    model.train()  # Set model to training mode
    total_loss = 0

    for batch in train_loader:
        user_input = batch['user']
        item_input = batch['item']
        target = batch['rating']

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(user_input, item_input)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    """
    Validate the model on the validation dataset.

    :param model: PyTorch model to validate.
    :param val_loader: DataLoader containing validation data.
    :param criterion: Loss function.
    :return: Average validation loss.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            user_input = batch['user']
            item_input = batch['item']
            target = batch['rating']

            # Forward pass
            output = model(user_input, item_input)

            # Compute loss
            loss = criterion(output, target)

            # Accumulate total loss
            total_loss += loss.item()

    return total_loss / len(val_loader)
