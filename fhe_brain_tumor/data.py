"""Data loading and preprocessing module for Brain Tumor MRI dataset."""
import os
import json
import random
import logging
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from .config import CONFIG, get_class_mapping, ensure_directories

logger = logging.getLogger(__name__)

def seed_worker(worker_id: int) -> None:
    """Ensure worker processes have reproducible seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int = CONFIG.seed) -> None:
    """Set random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_base_transform() -> transforms.Compose:
    """Standard transform: Grayscale, resize to target size, scale to [0, 1] via ToTensor only."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),  # Strictly scales [0, 255] pixels to [0.0, 1.0] range
    ])

def get_augmented_transform() -> transforms.Compose:
    """Light augmentation: rotation (+-10 deg), subtle translation (+-5%), then resize & ToTensor."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Resize((CONFIG.image_size, CONFIG.image_size)),
        transforms.ToTensor(),
    ])

def get_class_distribution(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """Measure real per-class sample counts in an ImageFolder dataset."""
    counts = {cls_name: 0 for cls_name in dataset.classes}
    for _, class_idx in dataset.samples:
        cls_name = dataset.classes[class_idx]
        counts[cls_name] += 1
    return counts

def load_datasets(use_augmentation: bool = False) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """
    Load raw ImageFolder datasets for Training and Testing.
    Testing dataset ALWAYS uses deterministic base transforms.
    """
    if not os.path.isdir(CONFIG.training_dir) or not os.path.isdir(CONFIG.testing_dir):
        raise FileNotFoundError(
            f"Dataset directories not found. Expected {CONFIG.training_dir} and {CONFIG.testing_dir}"
        )
    
    train_transform = get_augmented_transform() if use_augmentation else get_base_transform()
    test_transform = get_base_transform()
    
    train_dataset = datasets.ImageFolder(root=CONFIG.training_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=CONFIG.testing_dir, transform=test_transform)
    
    # Validate and assert class mapping
    expected_mapping = get_class_mapping()
    actual_mapping = train_dataset.class_to_idx
    if actual_mapping != expected_mapping:
        raise AssertionError(
            f"Class mapping mismatch: expected {expected_mapping}, but got {actual_mapping}"
        )
    
    # Save verified class mapping
    ensure_directories()
    with open(CONFIG.class_mapping_file, "w") as f:
        json.dump(actual_mapping, f, indent=2)
    logger.info("Saved verified class mapping to %s", CONFIG.class_mapping_file)
    
    return train_dataset, test_dataset

def create_stratified_split(
    dataset: datasets.ImageFolder,
    val_split: float = CONFIG.val_split,
    seed: int = CONFIG.seed
) -> Tuple[Subset, Subset]:
    """
    Create a stratified train/validation split from Training dataset.
    Ensures identical class ratios in both partitions.
    """
    targets = [target for _, target in dataset.samples]
    indices = list(range(len(dataset)))
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=targets,
        random_state=seed
    )
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    return train_subset, val_subset

def get_dataloaders(
    batch_size: int = CONFIG.batch_size,
    use_augmentation: bool = False,
    seed: int = CONFIG.seed
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Dict[str, int]]]:
    """
    Return reproducible (train_loader, val_loader, test_loader) alongside real sample counts.
    Note: Testing loader is preserved untouched for final evaluation only.
    """
    set_seed(seed)
    train_dataset, test_dataset = load_datasets(use_augmentation=use_augmentation)
    train_counts = get_class_distribution(train_dataset)
    test_counts = get_class_distribution(test_dataset)
    
    # Split Training into train and val
    train_subset, val_subset = create_stratified_split(
        train_dataset,
        val_split=CONFIG.val_split,
        seed=seed
    )
    
    # Build DataLoader generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=CONFIG.num_workers
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=CONFIG.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=CONFIG.num_workers
    )
    
    counts_report = {
        "training": train_counts,
        "testing": test_counts,
        "train_split": len(train_subset),
        "val_split": len(val_subset),
        "test_total": len(test_dataset)
    }
    
    logger.info("Dataset loaded successfully: %s", counts_report)
    return train_loader, val_loader, test_loader, counts_report
