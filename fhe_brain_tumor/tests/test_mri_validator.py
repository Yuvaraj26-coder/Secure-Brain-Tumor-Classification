"""Tests for Brain MRI Scan identification and Out-of-Distribution validation."""
import os
import numpy as np
from PIL import Image
import pytest
from fhe_brain_tumor.config import CONFIG
from fhe_brain_tumor.mri_validator import verify_brain_mri_scan

def test_authentic_mri_identification():
    """Verify that genuine Brain MRI scans from the dataset pass validation."""
    for cls_name in CONFIG.class_names:
        cls_dir = os.path.join(CONFIG.testing_dir, cls_name)
        if os.path.isdir(cls_dir):
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)[:3]]
            for file_path in files:
                result = verify_brain_mri_scan(file_path)
                assert result["is_valid_mri"] is True, f"Failed for real MRI {file_path}: {result['reasons']}"
                assert result["confidence"] >= 0.65
                assert result["verdict"] == "VALID_BRAIN_MRI"

def test_reject_color_photo():
    """Verify that colored / natural images are detected and rejected."""
    # Synthetic RGB color photo with high saturation
    color_arr = np.zeros((150, 150, 3), dtype=np.uint8)
    color_arr[:, :, 0] = 200  # Red
    color_arr[:, :, 1] = 50   # Green
    color_arr[:, :, 2] = 30   # Blue
    img = Image.fromarray(color_arr)
    
    result = verify_brain_mri_scan(img)
    assert result["is_valid_mri"] is False
    assert result["verdict"] == "REJECTED_NON_MRI"
    assert any("color" in r.lower() or "chroma" in r.lower() for r in result["reasons"])

def test_reject_white_background_document():
    """Verify that images with bright white borders (documents, scans, daylight photos) are rejected."""
    # White page with some dark lines in the middle
    doc_arr = np.ones((200, 200), dtype=np.uint8) * 255
    doc_arr[60:70, 30:170] = 20
    doc_arr[90:100, 30:170] = 20
    img = Image.fromarray(doc_arr)
    
    result = verify_brain_mri_scan(img)
    assert result["is_valid_mri"] is False
    assert result["verdict"] == "REJECTED_NON_MRI"
    assert any("periphery" in r.lower() or "border" in r.lower() for r in result["reasons"])

def test_reject_flat_or_blank_image():
    """Verify that uniform / blank images without contrast are rejected."""
    blank_arr = np.zeros((100, 100), dtype=np.uint8)
    img = Image.fromarray(blank_arr)
    
    result = verify_brain_mri_scan(img)
    assert result["is_valid_mri"] is False
    assert result["verdict"] == "REJECTED_NON_MRI"

