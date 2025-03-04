import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def train_model_single_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler,  # Kept for compatibility but no longer used
    grad_clip,
):
    """
    Trains the model for a single epoch.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (loss function): Loss function to optimize.
        optimizer (torch.optim): Optimizer for model training.
        device (torch.device): Device (CPU/GPU) to train on.
        scaler (torch.cuda.amp.GradScaler): Kept for compatibility but not used.
        grad_clip (float): Maximum gradient norm for clipping.

    Returns:
        tuple: Training loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for batch in progress_bar:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass without autocast
        outputs = model(images)

        if isinstance(outputs, dict):
                outputs = outputs['logits']
                
        loss = criterion(outputs, labels)

        # Backward pass and optimization without scaler
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            # Gradient clipping without unscaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({"loss": loss.item()})

    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    return train_loss, train_accuracy


def validate_model_single_epoch(
    model,
    val_loader,
    criterion,
    device,
):
    """
    Validates the model for a single epoch.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss function): Loss function for evaluation.
        device (torch.device): Device (CPU/GPU) to run validation on.

    Returns:
        tuple: Validation loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_correct = 0  # Track total correct predictions
    total_samples = 0  # Track total samples

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass without autocast
            outputs = model(images)

            if isinstance(outputs, dict):
                outputs = outputs['logits']

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_samples  
    return val_loss, val_accuracy


def train_model_single_epoch_multimodal(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler,  # Kept for compatibility but no longer used
    grad_clip,
):
    """
    Trains a multimodal model (image + text) for a single epoch.

    Args:
        model (nn.Module): Multimodal model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (loss function): Loss function to optimize.
        optimizer (torch.optim): Optimizer for training.
        device (torch.device): Device (CPU/GPU) to train on.
        scaler (torch.cuda.amp.GradScaler): Kept for compatibility but not used.
        grad_clip (float): Maximum gradient norm for clipping.

    Returns:
        tuple: Training loss and accuracy.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for batch in progress_bar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_masks = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass without autocast
        outputs = model(images, input_ids, attention_masks)

        if isinstance(outputs, dict):
                outputs = outputs['logits']
                
        loss = criterion(outputs, labels)

        # Backward pass and optimization without scaler
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            # Gradient clipping without unscaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({"loss": loss.item()})

    train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples
    return train_loss, train_accuracy


def validate_model_single_epoch_multimodal(
    model,
    val_loader,
    criterion,
    device,
):
    """
    Validates a multimodal model (image + text) for a single epoch.

    Args:
        model (nn.Module): The trained multimodal model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (loss function): Loss function for evaluation.
        device (torch.device): Device (CPU/GPU) to run validation on.

    Returns:
        tuple: Validation loss and accuracy.
    """
    model.eval()
    total_loss = 0
    total_correct = 0  # Track total correct predictions
    total_samples = 0  # Track total samples

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass without autocast
            outputs = model(images, input_ids, attention_masks)

            if isinstance(outputs, dict):
                outputs = outputs['logits']

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = total_correct / total_samples  
    return val_loss, val_accuracy


def save_checkpoint(epoch, model, optimizer, history, checkpoint_dir):
    """
    Saves a model checkpoint.

    Args:
        epoch (int): Current training epoch.
        model (nn.Module): Model to save.
        optimizer (torch.optim): Optimizer state to save.
        history (dict): Training history.
        checkpoint_dir (str): Directory to save the checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")


def get_device():
    """
    Determines the best available device (GPU, MPS, or CPU).

    Returns:
        torch.device: The best available device.
    """
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    return device


def evaluate_on_test(model, test_loader, device, class_names):
    """
    Evaluates the model on the test dataset and prints a classification report.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device (CPU/GPU) for evaluation.
        class_names (list): List of class names.

    Returns:
        None
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    print("Classification Report on Test Set:")
    print(classification_report(all_targets, all_preds, target_names=class_names))


def evaluate_on_test_multimodal(model, test_loader, device, class_names):
    """
    Evaluates a multimodal model (image + text) on the test dataset and prints a classification report.

    Args:
        model (nn.Module): The trained multimodal model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device (CPU/GPU) for evaluation.
        class_names (list): List of class names.

    Returns:
        None
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, input_ids, attention_masks)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    print("Classification Report on Test Set:")
    print(classification_report(all_targets, all_preds, target_names=class_names))


def plot_history(history, metric):
    """
    Plots training and validation loss or accuracy over epochs.

    Args:
        history (dict): Dictionary containing training/validation metrics.
        metric (str): Metric to plot ('loss' or 'accuracy').

    Raises:
        ValueError: If the metric is not 'loss' or 'accuracy'.
    """
    if metric == 'loss':
        train_metric = history['train_loss']
        val_metric = history['val_loss']
    elif metric == 'accuracy':
        train_metric = history['train_accuracy']
        val_metric = history['val_accuracy']
    else:
        raise ValueError("Unsupported metric. Please use 'loss' or 'accuracy'.")

    num_epochs = len(train_metric)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_metric, label=f'Train {metric}', marker='o')
    plt.plot(range(1, num_epochs + 1), val_metric, label=f'Validation {metric}', marker='o')
    plt.title(f'Train and validation {metric} over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(1, num_epochs + 1))
    plt.legend()
    plt.show()