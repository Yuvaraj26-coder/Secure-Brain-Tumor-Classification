"""Comprehensive evaluation module: metrics, confusion matrices, tumor miss rates, and FHE verification."""
import os
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CONFIG, ensure_directories
from .model import FHECompatibleCNN
from .data import get_dataloaders, load_datasets
from .client import Client
from .server import Server

logger = logging.getLogger(__name__)

def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence standard normal quantile
    p = successes / total
    denominator = 1 + (z**2) / total
    centre = (p + (z**2) / (2 * total)) / denominator
    spread = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)) / denominator
    lower = max(0.0, float(centre - spread))
    upper = min(1.0, float(centre + spread))
    return lower, upper

def evaluate_plaintext_test_set(
    model: nn.Module,
    device: torch.device = torch.device("cpu"),
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained model thoroughly on full held-out test partition.
    Generates precision, recall, F1-score, tumor-miss rate, and confusion matrix.
    """
    _, _, test_loader, counts = get_dataloaders()
    model.eval()
    
    all_preds = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            if isinstance(model, FHECompatibleCNN):
                logits, _ = model(images)
            else:
                logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate global and per-class metrics
    total = len(all_targets)
    correct = int((all_preds == all_targets).sum())
    accuracy = correct / total
    ci_lower, ci_upper = wilson_score_interval(correct, total)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=list(range(CONFIG.num_classes)), zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Specific tumor miss rate: False negative rate for each tumor class
    # For a tumor class: Miss Rate = 1.0 - Recall
    tumor_miss_rates = {}
    for idx, name in enumerate(CONFIG.class_names):
        rec = float(recall[idx])
        miss_rate = 1.0 - rec
        tumor_miss_rates[name] = {
            "recall": rec,
            "miss_rate": miss_rate,
            "miss_percentage": f"{miss_rate * 100:.2f}%",
            "support": int(support[idx])
        }
        
    metrics = {
        "total_samples": total,
        "accuracy": float(accuracy),
        "accuracy_ci_95": [ci_lower, ci_upper],
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.sum(f1 * support) / total),
        "per_class": {
            CONFIG.class_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            } for i in range(CONFIG.num_classes)
        },
        "tumor_miss_rates": tumor_miss_rates,
        "confusion_matrix": cm.tolist()
    }
    
    ensure_directories()
    # Save confusion matrix plot
    if save_plots:
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CONFIG.class_names,
            yticklabels=CONFIG.class_names
        )
        plt.title(f"{model.__class__.__name__} Test Confusion Matrix (Acc: {accuracy:.3f})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plot_path = os.path.join(CONFIG.plots_dir, f"{model.__class__.__name__.lower()}_confusion_matrix.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        metrics["confusion_matrix_plot"] = plot_path
        
    return metrics

def evaluate_encrypted_inference_subset(
    model: FHECompatibleCNN,
    subset_size: int = CONFIG.eval_subset_size,
    seed: int = CONFIG.seed,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Evaluate encrypted client-server inference on a stratified test subset.
    Measures:
      - Plaintext vs Encrypted accuracy on identical subset
      - Agreement rate between plaintext and encrypted predictions
      - Logit MAE, max absolute error, and per-class error distribution
      - 95% Wilson confidence intervals
    """
    logger.info("Starting encrypted inference evaluation on stratified subset (N=%d)...", subset_size)
    _, test_dataset = load_datasets()
    
    # Form stratified subset across test set
    targets = [target for _, target in test_dataset.samples]
    indices = np.arange(len(test_dataset))
    
    # Stratified selection
    per_class_quota = subset_size // CONFIG.num_classes
    selected_indices = []
    for cls_idx in range(CONFIG.num_classes):
        cls_indices = indices[np.array(targets) == cls_idx]
        np.random.seed(seed)
        chosen = np.random.choice(cls_indices, size=per_class_quota, replace=False)
        selected_indices.extend(chosen.tolist())
        
    client = Client()
    server = Server(model=model, public_context=client.public_context)
    
    plaintext_preds = []
    encrypted_preds = []
    true_labels = []
    logit_diffs = []
    per_class_errors = {cls_name: [] for cls_name in CONFIG.class_names}
    
    model.eval()
    for idx in selected_indices:
        img_path, true_cls = test_dataset.samples[idx]
        true_labels.append(true_cls)
        
        # 1. Plaintext baseline inference
        img_tensor = client.preprocess_image(img_path).unsqueeze(0).to(device)
        with torch.no_grad():
            pt_logits, _ = model(img_tensor)
            pt_pred = int(torch.argmax(pt_logits, dim=1).item())
            pt_logits_list = pt_logits[0].cpu().numpy().tolist()
            plaintext_preds.append(pt_pred)
            
        # 2. Encrypted Client-Server Inference
        cipher_bytes, windows_nb, _ = client.encrypt_image(img_path)
        resp_bytes, _ = server.process_request(cipher_bytes, windows_nb)
        enc_pred, enc_logits_list = client.decrypt_prediction(resp_bytes)
        encrypted_preds.append(enc_pred)
        
        # Error computation
        diff = np.abs(np.array(pt_logits_list) - np.array(enc_logits_list))
        logit_diffs.append(diff)
        per_class_errors[CONFIG.class_names[true_cls]].append(float(np.max(diff)))
        
    plaintext_preds = np.array(plaintext_preds)
    encrypted_preds = np.array(encrypted_preds)
    true_labels = np.array(true_labels)
    logit_diffs = np.array(logit_diffs)
    
    total = len(true_labels)
    pt_correct = int((plaintext_preds == true_labels).sum())
    enc_correct = int((encrypted_preds == true_labels).sum())
    agreed = int((plaintext_preds == encrypted_preds).sum())
    
    pt_acc = pt_correct / total
    enc_acc = enc_correct / total
    agreement_rate = agreed / total
    
    pt_ci = wilson_score_interval(pt_correct, total)
    enc_ci = wilson_score_interval(enc_correct, total)
    agree_ci = wilson_score_interval(agreed, total)
    
    mae = float(np.mean(logit_diffs))
    max_err = float(np.max(logit_diffs))
    per_class_mae = {cls: float(np.mean(errs)) for cls, errs in per_class_errors.items()}
    
    results = {
        "subset_size": total,
        "per_class_sample_count": per_class_quota,
        "plaintext_accuracy": pt_acc,
        "plaintext_acc_ci95": list(pt_ci),
        "encrypted_accuracy": enc_acc,
        "encrypted_acc_ci95": list(enc_ci),
        "prediction_agreement_rate": agreement_rate,
        "agreement_ci95": list(agree_ci),
        "accuracy_degradation": float(pt_acc - enc_acc),
        "mean_absolute_logit_error": mae,
        "max_absolute_logit_error": max_err,
        "per_class_mae": per_class_mae
    }
    
    ensure_directories()
    with open(os.path.join(CONFIG.results_dir, "encrypted_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info("Encrypted inference evaluation complete: Acc=%.4f (Agree=%.4f, MaxErr=%.6f)", enc_acc, agreement_rate, max_err)
    return results
