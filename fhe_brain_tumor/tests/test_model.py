"""Tests for model architectures, FHE compliance, and linear pooling folding."""
import pytest
import torch
import torch.nn as nn
from fhe_brain_tumor.config import CONFIG
from fhe_brain_tumor.model import (
    FHECompatibleCNN,
    ReLUBaselineCNN,
    build_avgpool_folding_matrix,
    inspect_model_fhe_compliance
)

def test_fhe_model_forward_shape():
    """Verify output tensor has shape (batch_size, 4)."""
    model = FHECompatibleCNN()
    x = torch.randn(2, 1, CONFIG.image_size, CONFIG.image_size)
    out, max_act = model(x, track_max_act=True)
    assert out.shape == (2, CONFIG.num_classes)
    assert max_act >= 0.0

def test_relu_model_forward_shape():
    """Verify baseline model output has shape (batch_size, 4)."""
    model = ReLUBaselineCNN()
    x = torch.randn(2, 1, CONFIG.image_size, CONFIG.image_size)
    out = model(x)
    assert out.shape == (2, CONFIG.num_classes)

def test_zero_non_polynomial_operations():
    """Verify strictly zero non-polynomial operations exist in FHECompatibleCNN."""
    model = FHECompatibleCNN()
    compliance = inspect_model_fhe_compliance(model)
    assert compliance["is_fhe_compliant"] is True
    assert len(compliance["non_poly_violations"]) == 0
    assert compliance["multiplicative_depth"] == 5

def test_avgpool_folding_mathematical_equivalence():
    """
    Formally verify that folding Average Pooling into FC1 produces mathematically identical
    results to standard Sequential(AvgPool, Flatten, FC1) within floating point tolerance.
    """
    model = FHECompatibleCNN()
    model.eval()
    
    # Feature map after conv and square: shape (1, 4, 8, 8)
    conv_feat = torch.randn(1, CONFIG.conv_out_channels, CONFIG.conv_output_spatial, CONFIG.conv_output_spatial)
    
    # 1. Standard pipeline: AvgPool -> Flatten -> FC1
    with torch.no_grad():
        pooled = model.avg_pool(conv_feat) # (1, 4, 4, 4)
        flat = pooled.view(1, -1)          # (1, 64)
        std_out = model.fc1(flat)          # (1, 32)
        
    # 2. Folded pipeline: W_folded = W_fc1 @ P
    P = build_avgpool_folding_matrix()     # (64, 256)
    W_folded = model.fc1.weight.data @ P   # (32, 256)
    conv_flat = conv_feat.view(1, -1)      # (1, 256)
    folded_out = conv_flat @ W_folded.T + model.fc1.bias.data # (1, 32)
    
    # Assert exact numerical equivalence
    max_diff = (std_out - folded_out).abs().max().item()
    assert max_diff < 1e-6, f"Folding discrepancy too large: {max_diff}"
