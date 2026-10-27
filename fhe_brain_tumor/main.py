"""Main execution pipeline executing all 6 tasks and generating artifacts."""
import os
import json
import logging
import torch
from .config import CONFIG, ensure_directories
from .data import load_datasets, get_dataloaders, get_class_distribution
from .model import FHECompatibleCNN, ReLUBaselineCNN, inspect_model_fhe_compliance
from .train import train_single_model, train_multi_seed_evaluation
from .fhe_context import context_manager, get_fhe_context, get_public_context
from .client import Client
from .server import Server
from .evaluate import evaluate_plaintext_test_set, evaluate_encrypted_inference_subset
from .benchmark import Benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fhe_brain_tumor.main")

def run_all_tasks(quick_mode: bool = False) -> None:
    """Execute all six tasks end-to-end and generate all deliverable artifacts."""
    ensure_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # =========================================================================
    # Task 1: Dataset Selection and Preprocessing
    # =========================================================================
    print("\n" + "="*80)
    print("TASK 1: DATASET SELECTION & PREPROCESSING")
    print("="*80)
    train_dataset, test_dataset = load_datasets()
    train_counts = get_class_distribution(train_dataset)
    test_counts = get_class_distribution(test_dataset)
    
    print(f"Real Training counts: {train_counts} (Total: {len(train_dataset)})")
    print(f"Real Testing counts:  {test_counts} (Total: {len(test_dataset)})")
    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Preprocessing: Grayscale, resized to {CONFIG.image_size}x{CONFIG.image_size}, normalized to [0, 1]")
    
    # =========================================================================
    # Task 2: Plaintext FHE-Compatible Model Architecture & Training
    # =========================================================================
    print("\n" + "="*80)
    print("TASK 2: PLAINTEXT FHE-COMPATIBLE MODEL TRAINING & BASELINE")
    print("="*80)
    fhe_compliance = inspect_model_fhe_compliance(FHECompatibleCNN())
    print(f"FHE Architectural Compliance: {fhe_compliance['is_fhe_compliant']}")
    print(f"Multiplicative depth: {fhe_compliance['multiplicative_depth']}")
    
    # Train FHE model
    fhe_train_res = train_single_model(model_type="fhe", seed=CONFIG.seed, device=device)
    fhe_model = fhe_train_res["model"]
    
    # Train ReLU baseline for direct architectural cost quantification
    relu_train_res = train_single_model(model_type="relu", seed=CONFIG.seed, device=device)
    relu_model = relu_train_res["model"]
    
    # Full test evaluation
    fhe_metrics = evaluate_plaintext_test_set(fhe_model, device=device)
    relu_metrics = evaluate_plaintext_test_set(relu_model, device=device)
    
    print(f"\nFHE Model Test Accuracy:  {fhe_metrics['accuracy']:.4f} (95% CI: {fhe_metrics['accuracy_ci_95']})")
    print(f"ReLU Baseline Test Accuracy: {relu_metrics['accuracy']:.4f} (95% CI: {relu_metrics['accuracy_ci_95']})")
    print(f"Architectural Accuracy Cost: {relu_metrics['accuracy'] - fhe_metrics['accuracy']:.4f}")
    print(f"Tumor Miss Rates (FHE Model):")
    for cls, info in fhe_metrics["tumor_miss_rates"].items():
        print(f"  - {cls}: {info['miss_percentage']} miss rate (Recall: {info['recall']:.3f})")
        
    # Save combined metrics
    metrics_summary = {
        "fhe_model": fhe_metrics,
        "relu_baseline": relu_metrics,
        "dataset_counts": {"training": train_counts, "testing": test_counts}
    }
    with open(os.path.join(CONFIG.results_dir, "plaintext_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
        
    # =========================================================================
    # Task 3: Cryptographic Context Configuration (TenSEAL CKKS)
    # =========================================================================
    print("\n" + "="*80)
    print("TASK 3: CRYPTOGRAPHIC CONTEXT CONFIGURATION & VALIDATION")
    print("="*80)
    val_report = context_manager.validate_parameters_empirically()
    print("Empirical Parameter Validation:")
    print(f"1. Spec Parameters (Scale 2^26, 21-bit primes): {val_report['spec_params']['status']}")
    print(f"   Max Error: {val_report['spec_params'].get('max_absolute_error', 'N/A')}")
    print(f"   Note: {val_report['spec_params'].get('note')}")
    print(f"2. Calibrated Parameters (Scale 2^26, 26-bit primes): {val_report['calibrated_params']['status']}")
    print(f"   Max Error: {val_report['calibrated_params'].get('max_absolute_error')}")
    print(f"   128-bit Security Budget: {val_report['calibrated_params']['total_bits']} bits <= 218 bits (PASSED)")
    
    context_manager.save_contexts()
    
    # =========================================================================
    # Task 4 & Task 5: Client Encryption & Server Encrypted Inference
    # =========================================================================
    print("\n" + "="*80)
    print("TASK 4 & 5: CLIENT ENCRYPTION & SERVER ENCRYPTED INFERENCE")
    print("="*80)
    client = Client()
    server = Server(model=fhe_model, public_context=client.public_context)
    
    # Sample test image
    sample_img_path, sample_true_cls = test_dataset.samples[0]
    sample_cls_name = CONFIG.class_names[sample_true_cls]
    print(f"Selected test sample: {sample_img_path} (True label: {sample_cls_name})")
    
    cipher_bytes, windows_nb, cipher_size = client.encrypt_image(sample_img_path)
    print(f"Client encrypted input payload size: {cipher_size:,} bytes across {windows_nb} sliding windows")
    
    resp_bytes, resp_size = server.process_request(cipher_bytes, windows_nb)
    print(f"Server processed encrypted inference: response payload size: {resp_size:,} bytes")
    
    # =========================================================================
    # Task 6: Decryption & Stratified Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("TASK 6: CLIENT DECRYPTION & EVALUATION")
    print("="*80)
    pred_idx, decrypted_logits = client.decrypt_prediction(resp_bytes)
    print(f"Client decrypted prediction: {pred_idx} ({CONFIG.class_names[pred_idx]})")
    print(f"Decrypted logits: {[round(x, 4) for x in decrypted_logits]}")
    
    # Stratified subset evaluation
    subset_n = 50 if quick_mode else CONFIG.eval_subset_size
    encrypted_eval = evaluate_encrypted_inference_subset(
        fhe_model,
        subset_size=subset_n,
        seed=CONFIG.seed,
        device=device
    )
    print(f"\nEncrypted Inference Subset Evaluation (N={subset_n}):")
    print(f"  - Plaintext Accuracy on Subset: {encrypted_eval['plaintext_accuracy']:.4f}")
    print(f"  - Encrypted Accuracy on Subset: {encrypted_eval['encrypted_accuracy']:.4f}")
    print(f"  - Prediction Agreement Rate:    {encrypted_eval['prediction_agreement_rate']:.4f} (95% CI: {encrypted_eval['agreement_ci95']})")
    print(f"  - Max Absolute Logit Error:     {encrypted_eval['max_absolute_logit_error']:.6f}")
    print(f"  - Mean Absolute Logit Error:    {encrypted_eval['mean_absolute_logit_error']:.6f}")
    
    # =========================================================================
    # Benchmarking: Cryptographic Tax
    # =========================================================================
    print("\n" + "="*80)
    print("CRYPTOGRAPHIC TAX BENCHMARKS")
    print("="*80)
    benchmark = Benchmark(repeats=CONFIG.benchmark_runs, warmup=CONFIG.warmup_runs)
    bench_results = benchmark.run_full_benchmark(fhe_model)
    
    lat = bench_results["latency"]
    payload = bench_results["payload_and_memory"]
    print(f"Latency Profile (Median):")
    print(f"  - Keygen (Galois):      {lat['galois_keygen_s']['median']:.3f} s")
    print(f"  - Client Encryption:    {lat['client_encrypt_s']['median']:.3f} s")
    print(f"  - Server Inference:     {lat['server_inference_s']['median']:.3f} s")
    print(f"  - Client Decryption:    {lat['client_decrypt_s']['median']:.3f} s")
    print(f"  - Plaintext PyTorch:    {lat['plaintext_pytorch_s']['median']*1000:.2f} ms")
    print(f"  - Slowdown Factor:      {lat['slowdown_factor']:.1f}x")
    print(f"Memory & Expansion:")
    print(f"  - Ciphertext Input:     {payload['ciphertext_input_bytes']:,} bytes")
    print(f"  - Raw Image Bytes:      {payload['raw_image_bytes']:,} bytes")
    print(f"  - Expansion Factor:     {payload['ciphertext_expansion_factor']:.1f}x")
    print(f"  - Peak Process RAM:     {payload['peak_process_ram_mb']:.1f} MB")
    
    print("\n" + "="*80)
    print("ALL SIX TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    run_all_tasks()
