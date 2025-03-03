import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    return device


def evaluate_on_test(model, test_loader, device, class_names):
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