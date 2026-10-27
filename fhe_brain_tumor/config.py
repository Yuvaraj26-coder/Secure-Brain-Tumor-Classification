"""Configuration module for FHE Brain Tumor Classification project."""
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class Config:
    """Frozen configuration containing all hyperparameters, FHE parameters, and paths."""
    
    # Dataset Parameters
    data_root: str = "data"
    training_dir: str = "data/Training"
    testing_dir: str = "data/Testing"
    image_size: int = 28  # 28x28 justified for CKKS 4096-slot constraint and 8x8->4x4 pooling
    num_classes: int = 4
    class_names: Tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
    val_split: float = 0.1
    seed: int = 42
    multi_seeds: Tuple[int, ...] = (42, 43, 44)
    batch_size: int = 32
    num_workers: int = 0
    
    # Model Architecture: Conv -> square -> AvgPool -> flatten -> FC1 -> square -> FC2 (4 logits)
    conv_in_channels: int = 1
    conv_out_channels: int = 4
    conv_kernel_size: int = 7
    conv_stride: int = 3
    # Spatial calculation: (28 - 7) // 3 + 1 = 8
    conv_output_spatial: int = 8
    avg_pool_kernel_size: int = 2
    avg_pool_stride: int = 2
    # Spatial after pool: 8 // 2 = 4
    pool_output_spatial: int = 4
    # Flattened dimension: 4 channels * 4 * 4 = 64
    flattened_dim: int = 64
    fc_hidden_dim: int = 32
    weight_init_std: float = 0.02
    
    # Training Parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 6
    grad_clip_norm: float = 1.0
    
    # FHE Specification Parameters (Task 3 Spec)
    poly_modulus_degree: int = 8192  # N=8192 provides 4096 slots
    spec_coeff_mod_bit_sizes: Tuple[int, ...] = (40, 21, 21, 21, 21, 21, 21, 40)
    spec_global_scale: int = 2**26
    
    # FHE Calibrated / Minimal Justified Alternative Parameters
    # Alternative 1: Matched 26-bit primes, exactly 218 bits total budget (128-bit security for N=8192)
    calibrated_coeff_mod_bit_sizes: Tuple[int, ...] = (31, 26, 26, 26, 26, 26, 26, 31)
    calibrated_global_scale: int = 2**26
    
    # Alternative 2: Keep 21-bit primes, match scale to 2**21 (206 bits total budget, 128-bit security)
    alt21_coeff_mod_bit_sizes: Tuple[int, ...] = (40, 21, 21, 21, 21, 21, 21, 40)
    alt21_global_scale: int = 2**21
    
    # Artifact Paths
    artifacts_dir: str = "artifacts"
    results_dir: str = "artifacts/results"
    plots_dir: str = "artifacts/plots"
    model_checkpoint: str = "artifacts/best_fhe_model.pt"
    relu_checkpoint: str = "artifacts/best_relu_model.pt"
    class_mapping_file: str = "artifacts/class_mapping.json"
    benchmark_file: str = "artifacts/results/benchmarks.json"
    results_csv: str = "artifacts/results/evaluation_metrics.csv"
    summary_pdf_path: str = "summary.pdf"
    
    # Evaluation & Benchmarking
    eval_subset_size: int = 200  # Stratified test subset for FHE evaluation
    benchmark_runs: int = 5
    warmup_runs: int = 2
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.image_size not in [28, 32]:
            raise ValueError(f"image_size must be 28 or 32, got {self.image_size}")
        if not (0 < self.val_split < 1):
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match class_names length")

CONFIG = Config()

def get_class_mapping() -> Dict[str, int]:
    """Return the expected class-to-index mapping."""
    return {name: idx for idx, name in enumerate(CONFIG.class_names)}

def ensure_directories() -> None:
    """Create all necessary output directories."""
    for d in [CONFIG.artifacts_dir, CONFIG.results_dir, CONFIG.plots_dir]:
        os.makedirs(d, exist_ok=True)
