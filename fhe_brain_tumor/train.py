"""Training pipeline for FHE-compatible model and ReLU baseline with stability safeguards."""
import os
import json
import logging
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .config import CONFIG, ensure_directories, get_class_mapping
from .model import FHECompatibleCNN, ReLUBaselineCNN
from .data import get_dataloaders, set_seed

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops training when validation accuracy ceases improvement."""
    
    def __init__(self, patience: int = CONFIG.patience, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_acc = 0.0

    def __call__(self, val_acc: float) -> bool:
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.best_acc = val_acc
            return True
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.best_acc = val_acc
            self.counter = 0
            return True

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    is_fhe_model: bool = True
) -> Tuple[float, float, float]:
    """Train single epoch with gradient clipping and activation tracking."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_max_act = 0.0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if is_fhe_model:
            logits, batch_max_act = model(images, track_max_act=True)
            epoch_max_act = max(epoch_max_act, batch_max_act)
        else:
            logits = model(images)
            
        loss = criterion(logits, targets)
        loss.backward()
        
        # Gradient clipping to stabilize square activation derivatives
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.grad_clip_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc, epoch_max_act

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_fhe_model: bool = True
) -> Tuple[float, float, float]:
    """Evaluate model on validation or test partition."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    eval_max_act = 0.0
    
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if is_fhe_model:
            logits, batch_max_act = model(images, track_max_act=True)
            eval_max_act = max(eval_max_act, batch_max_act)
        else:
            logits = model(images)
            
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc, eval_max_act

def train_single_model(
    model_type: str = "fhe",
    seed: int = CONFIG.seed,
    use_augmentation: bool = False,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Train a single model (FHE-compatible or ReLU baseline) with early stopping.
    """
    set_seed(seed)
    train_loader, val_loader, test_loader, counts = get_dataloaders(
        use_augmentation=use_augmentation,
        seed=seed
    )
    
    is_fhe = (model_type.lower() == "fhe")
    model = FHECompatibleCNN().to(device) if is_fhe else ReLUBaselineCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay
    )
    
    early_stopping = EarlyStopping(patience=CONFIG.patience)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "max_act": []}
    best_weights = None
    
    logger.info("Starting training %s model (seed=%d, aug=%s)", model_type, seed, use_augmentation)
    for epoch in range(1, CONFIG.max_epochs + 1):
        tr_loss, tr_acc, tr_max_act = train_epoch(model, train_loader, criterion, optimizer, device, is_fhe)
        val_loss, val_acc, val_max_act = evaluate_epoch(model, val_loader, criterion, device, is_fhe)
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["max_act"].append(tr_max_act)
        
        logger.info(
            "Epoch %02d/%02d - Loss: %.4f, Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f | Max |Act|: %.3f",
            epoch, CONFIG.max_epochs, tr_loss, tr_acc, val_loss, val_acc, tr_max_act
        )
        
        is_best = early_stopping(val_acc)
        if is_best:
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break
            
    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Evaluate strictly on held-out test partition
    test_loss, test_acc, _ = evaluate_epoch(model, test_loader, criterion, device, is_fhe)
    logger.info("Test set evaluation: Acc=%.4f, Loss=%.4f", test_acc, test_loss)
    
    # Save checkpoint
    checkpoint_path = CONFIG.model_checkpoint if is_fhe else CONFIG.relu_checkpoint
    ensure_directories()
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "image_size": CONFIG.image_size,
            "conv_channels": (CONFIG.conv_in_channels, CONFIG.conv_out_channels),
            "hidden_dim": CONFIG.fc_hidden_dim,
            "num_classes": CONFIG.num_classes
        },
        "class_mapping": get_class_mapping(),
        "best_val_acc": early_stopping.best_acc,
        "test_acc": test_acc,
        "history": history
    }, checkpoint_path)
    logger.info("Saved best checkpoint to %s", checkpoint_path)
    
    return {
        "model": model,
        "model_type": model_type,
        "seed": seed,
        "best_val_acc": early_stopping.best_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "checkpoint_path": checkpoint_path
    }

def train_multi_seed_evaluation(
    model_type: str = "fhe",
    seeds: Tuple[int, ...] = CONFIG.multi_seeds,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """Train model across >=3 seeds to report mean +- std for rigorous reporting."""
    results = []
    for s in seeds:
        res = train_single_model(model_type=model_type, seed=s, device=device)
        results.append(res["test_acc"])
        
    mean_acc = float(np.mean(results))
    std_acc = float(np.std(results))
    summary = {
        "model_type": model_type,
        "seeds": list(seeds),
        "test_accuracies": results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc
    }
    logger.info("Multi-seed summary for %s: %.4f +- %.4f", model_type, mean_acc, std_acc)
    return summary
