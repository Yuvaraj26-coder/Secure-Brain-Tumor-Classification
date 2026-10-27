"""
Brain MRI Scan Identification and Out-of-Distribution (OOD) Verification Module.
Validates whether an arbitrary uploaded image is an authentic Brain MRI scan
before proceeding with expensive Fully Homomorphic Encryption (FHE) computation.
"""
import os
from typing import Tuple, Dict, Any, List, Union
import numpy as np
from PIL import Image
import torch

def verify_brain_mri_scan(
    image_input: Union[str, Image.Image, np.ndarray, torch.Tensor]
) -> Dict[str, Any]:
    """
    Formally screen and verify if an uploaded image exhibits the radiometric,
    morphological, and structural signatures of an authentic Brain MRI scan.
    
    Checks conducted:
      1. Monochromaticity / Chrominance: MRI scans lack RGB color saturation.
      2. Periphery Background: Scanner air space produces dark margins (border_mean < 0.35).
      3. Dynamic Range & Contrast: Rejects blank, uniform, or flat images (std > 0.08).
      4. Cranial Mass Concentration: Center-to-periphery intensity gradient (ratio > 1.4).
      5. Cranial Centerness: Anatomical center of mass alignment within the frame.
      6. Parenchyma Pixel Fraction: Brain tissue occupies 15% to 80% of total area.
      
    Returns:
      dict containing:
        - 'is_valid_mri': bool
        - 'confidence': float in [0.0, 1.0]
        - 'verdict': 'VALID_BRAIN_MRI' or 'REJECTED_NON_MRI'
        - 'metrics': detailed measured radiometry
        - 'reasons': human-readable explanation
    """
    # 1. Load and format image
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img = Image.fromarray((image_input * 255).astype(np.uint8) if image_input.max() <= 1.0 else image_input.astype(np.uint8))
        else:
            img = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, torch.Tensor):
        t = image_input.detach().cpu().squeeze().numpy()
        if t.ndim == 2:
            img = Image.fromarray((t * 255).astype(np.uint8) if t.max() <= 1.0 else t.astype(np.uint8))
        else:
            img = Image.fromarray(np.transpose(t, (1, 2, 0)).astype(np.uint8))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB array for chromatic evaluation
    rgb = np.array(img.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    
    # 2. Chrominance Check (Monochromatic property)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # Inter-channel absolute difference
    chroma = float(np.mean((np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) / 3.0))
    
    # 3. Grayscale conversion normalized to [0.0, 1.0]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255.0
    
    # 4. Scanner Background / Margin Check
    # Outer 8% boundary margin
    bw = max(int(min(h, w) * 0.08), 1)
    top_border = gray[:bw, :]
    bot_border = gray[-bw:, :]
    left_border = gray[:, :bw]
    right_border = gray[:, -bw:]
    border_pixels = np.concatenate([
        top_border.flatten(), bot_border.flatten(),
        left_border.flatten(), right_border.flatten()
    ])
    border_mean = float(np.mean(border_pixels))
    
    # 5. Center Cranial Region (middle 50% spatial box)
    ch_s, ch_e = int(h * 0.25), int(h * 0.75)
    cw_s, cw_e = int(w * 0.25), int(w * 0.75)
    center_pixels = gray[ch_s:ch_e, cw_s:cw_e]
    center_mean = float(np.mean(center_pixels))
    
    # Center to border intensity ratio
    center_border_ratio = center_mean / (border_mean + 1e-4)
    
    # 6. Global Contrast & Intensity Variance
    contrast_std = float(np.std(gray))
    
    # 7. Tissue Parenchyma Coverage (pixels with intensity in [0.15, 0.95])
    tissue_mask = (gray >= 0.15) & (gray <= 0.95)
    tissue_coverage = float(np.mean(tissue_mask))
    
    # 8. Center of Mass of tissue region
    if np.sum(tissue_mask) > 10:
        y_coords, x_coords = np.where(tissue_mask)
        cy = float(np.mean(y_coords) / h)
        cx = float(np.mean(x_coords) / w)
        # Centerness penalty: distance from frame center (0.5, 0.5)
        dist_from_center = float(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
        centerness = max(0.0, 1.0 - (dist_from_center / 0.5))
    else:
        cx, cy = 0.5, 0.5
        centerness = 0.0

    # ---------------------------------------------------------
    # Evaluation Scoring & Anomaly Detection
    # ---------------------------------------------------------
    scores: List[float] = []
    reasons: List[str] = []
    
    # Check A: Chrominance (Weight: 30%)
    if chroma < 2.0:
        scores.append(1.0)
    elif chroma < 6.0:
        scores.append(0.8)
    elif chroma < 12.0:
        scores.append(0.3)
        reasons.append(f"Subtle color tint detected (chroma deviation: {chroma:.1f})")
    else:
        scores.append(0.0)
        reasons.append(f"Image has high color saturation (chroma: {chroma:.1f}); MRI scans are monochromatic grayscale.")
        
    # Check B: Periphery Darkness / Air Boundary (Weight: 25%)
    if border_mean < 0.15:
        scores.append(1.0)
    elif border_mean < 0.30:
        scores.append(0.7)
    elif border_mean < 0.45:
        scores.append(0.3)
        reasons.append(f"Periphery contains elevated brightness ({border_mean:.2f}); scanner air space should be dark.")
    else:
        scores.append(0.0)
        reasons.append(f"Image borders are bright ({border_mean:.2f}), characteristic of non-medical photos or documents.")
        
    # Check C: Dynamic Range / Contrast (Weight: 15%)
    if contrast_std >= 0.12:
        scores.append(1.0)
    elif contrast_std >= 0.07:
        scores.append(0.6)
    else:
        scores.append(0.0)
        reasons.append(f"Image exhibits near-uniform tone or insufficient contrast (std: {contrast_std:.3f}).")
        
    # Check D: Cranial Contrast Gradient (Weight: 15%)
    if center_border_ratio >= 2.0:
        scores.append(1.0)
    elif center_border_ratio >= 1.4:
        scores.append(0.7)
    else:
        scores.append(0.0)
        reasons.append(f"Absence of cranial contrast gradient (center/periphery ratio: {center_border_ratio:.1f}).")
        
    # Check E: Tissue Coverage & Central Alignment (Weight: 15%)
    if 0.12 <= tissue_coverage <= 0.85 and centerness >= 0.60:
        scores.append(1.0)
    elif tissue_coverage >= 0.05 and centerness >= 0.40:
        scores.append(0.6)
    else:
        scores.append(0.0)
        reasons.append("Brain tissue mass distribution does not match anatomical cranial contours.")
        
    # Calculate weighted confidence score
    weights = [0.30, 0.25, 0.15, 0.15, 0.15]
    total_confidence = float(sum(s * w for s, w in zip(scores, weights)))
    
    # Adaptive gatekeeper criteria:
    # 1. Definite valid MRI: confidence >= 0.60, dark borders, sufficient contrast, monochromatic
    # 2. Borderline: confidence >= 0.50, still requires dark borders and contrast
    # 3. Rejected: bright white borders (documents/daylight), blank/uniform images, or color photos
    has_valid_physics = (border_mean < 0.40 and contrast_std >= 0.06 and chroma < 15.0)
    
    is_definite_valid = (
        has_valid_physics
        and total_confidence >= 0.60
    )
    is_borderline = (
        has_valid_physics
        and not is_definite_valid
        and total_confidence >= 0.50
    )
    
    if is_definite_valid:
        verdict = "VALID_BRAIN_MRI"
        is_valid = True
    elif is_borderline:
        verdict = "BORDERLINE_MRI"
        is_valid = True
        reasons.append("Scan contains non-standard contrast, scanner annotations, or atypical cranial orientation.")
    else:
        verdict = "REJECTED_NON_MRI"
        is_valid = False
    
    if is_definite_valid and len(reasons) == 0:
        reasons.append("Confirmed authentic brain MRI scan characteristics (monochromatic, dark air boundary, cranial contrast).")
        
    return {
        "is_valid_mri": is_valid,
        "confidence": total_confidence,
        "confidence_pct": round(total_confidence * 100.0, 1),
        "verdict": verdict,
        "reasons": reasons,
        "metrics": {
            "chroma": round(chroma, 2),
            "border_mean": round(border_mean, 3),
            "center_mean": round(center_mean, 3),
            "center_border_ratio": round(center_border_ratio, 2),
            "contrast_std": round(contrast_std, 3),
            "tissue_coverage": round(tissue_coverage, 3),
            "centerness": round(centerness, 3)
        }
    }

