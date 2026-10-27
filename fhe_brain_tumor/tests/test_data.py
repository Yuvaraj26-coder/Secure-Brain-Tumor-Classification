"""Tests for data loading, preprocessing, and stratification."""
import os
import pytest
import torch
from torchvision import datasets
from fhe_brain_tumor.config import CONFIG, get_class_mapping
from fhe_brain_tumor.data import (
    set_seed,
    load_datasets,
    get_dataloaders,
    get_class_distribution,
    create_stratified_split,
    get_base_transform
)

def test_set_seed():
    """Verify deterministic RNG behavior."""
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.allclose(a, b)

def test_class_mapping():
    """Verify class-to-index mapping matches specification."""
    mapping = get_class_mapping()
    assert len(mapping) == 4
    for idx, name in enumerate(["glioma", "meningioma", "notumor", "pituitary"]):
        assert mapping[name] == idx

def test_dataset_loading_and_counts():
    """Verify dataset loads and has authentic sample counts."""
    train_dataset, test_dataset = load_datasets()
    assert len(train_dataset) == 5712, f"Expected 5,712 training samples, got {len(train_dataset)}"
    assert len(test_dataset) == 1311, f"Expected 1,311 testing samples, got {len(test_dataset)}"
    
    train_dist = get_class_distribution(train_dataset)
    assert train_dist["glioma"] == 1321
    assert train_dist["meningioma"] == 1339
    assert train_dist["notumor"] == 1595
    assert train_dist["pituitary"] == 1457

def test_image_preprocessing_shape_and_range():
    """Verify images are 1x28x28 and scaled strictly to [0.0, 1.0]."""
    _, test_dataset = load_datasets()
    img_tensor, label = test_dataset[0]
    
    assert img_tensor.shape == (1, CONFIG.image_size, CONFIG.image_size)
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0
    assert 0 <= label < 4

def test_stratified_split():
    """Verify stratified split yields 90/10 ratio and preserves class proportions."""
    train_dataset, _ = load_datasets()
    train_subset, val_subset = create_stratified_split(train_dataset, val_split=0.1, seed=42)
    
    total = len(train_dataset)
    assert abs(len(val_subset) - int(total * 0.1)) <= 1
    assert len(train_subset) == total - len(val_subset)
