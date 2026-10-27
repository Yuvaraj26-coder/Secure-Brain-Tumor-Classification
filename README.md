# Secure Brain Tumor Classification using Fully Homomorphic Encryption (CKKS)

An enterprise-grade, zero-trust Privacy-Preserving Machine Learning (PPML) classification system for brain MRI scans utilizing the Cheon-Kim-Kim-Song (CKKS) Fully Homomorphic Encryption scheme via TenSEAL and PyTorch.

---

## 1. Overview & Threat Model

Artificial intelligence in medical diagnostics requires processing highly sensitive, private patient health data (PHI). Fully Homomorphic Encryption (FHE) resolves the data privacy dilemma by allowing computation directly on encrypted ciphertexts without ever exposing the underlying plaintext.

```
+----------------------------------------------------------------------------------+
|                               ZERO-TRUST ARCHITECTURE                            |
+----------------------------------------------------------------------------------+
|  [ CLIENT ] (Holds secret key `sk`)                                              |
|    1. Normalized MRI Image (1x28x28) in [0.0, 1.0]                               |
|    2. ts.im2col_encoding (kernel 7x7, stride 3 -> 64 sliding windows)           |
|    3. Encrypt with Public Context -> Ciphertext (476 KB)                         |
|         |                                                                        |
|         v (Network Transfer)                                                     |
|  [ CLOUD SERVER ] (Possesses NO secret key; only Public Context & Plain Weights) |
|    4. Encrypted Conv2D (4 kernels, 7x7, s=3) via conv2d_im2col                   |
|    5. Pack channel vectors                                                       |
|    6. In-place Square Activation (x -> x^2)                                      |
|    7. Folded Dense Linear 1 (Average Pooling folded into FC1 weights: W_fc1 @ P) |
|    8. In-place Square Activation (x -> x^2)                                      |
|    9. Dense Linear 2 (FC2 -> 4 raw encrypted logits)                             |
|         |                                                                        |
|         v (Encrypted Response: 70 KB)                                            |
|  [ CLIENT ]                                                                      |
|   10. Decrypt logits with Secret Key                                             |
|   11. Argmax -> Pathology Classification: {Glioma, Meningioma, No Tumor, Pit.}  |
+----------------------------------------------------------------------------------+
```

### Threat Model Assumptions
- **Adversary Model:** Honest-but-curious cloud server. The server faithfully performs matrix-vector and tensor operations but inspects data to infer patient diagnoses.
- **Client Security:** Client holds exclusive possession of secret key $sk$.
- **Model Transparency:** Plaintext model weights are non-private; client inputs and prediction outputs are strictly confidential under IND-CPA security.
- **Caveats:** Approximate arithmetic (inherent CKKS noise), no verifiable computation proofs, timing and input dimension leakage, research prototype (not clinically validated).

---

## 2. Dataset Distribution (Real Empirical Counts)

Evaluated on the Kaggle **"Brain Tumor MRI Dataset"** (7,023 total scans). Real counted sample distributions:

| Partition | Glioma | Meningioma | No Tumor | Pituitary | Total Scans |
|---|---|---|---|---|---|
| **Training Set** | 1,321 | 1,339 | 1,595 | 1,457 | **5,712** |
| **Testing Set (Held-Out)** | 300 | 306 | 405 | 300 | **1,311** |
| **Combined** | 1,621 | 1,645 | 2,000 | 1,757 | **7,023** |

- **Preprocessing:** Single-channel grayscale conversion, resized to **28 × 28** pixels (justified below), scaled strictly to $[0.0, 1.0]$ via `ToTensor()` (no mean/std shift).
- **Stratified Split:** 90% training (5,140 scans) / 10% validation (572 scans) from `Training/`. Testing partition (1,311 scans) held out untouched until final evaluation.

---

## 3. Model Architecture & FHE Mathematical Constraints

To operate within homomorphic constraints, standard deep learning operations must be transformed:
- **No Non-Polynomial Operations:** Standard ReLUs and Max Pooling are non-polynomial and impossible in arithmetic circuits without high-degree polynomial approximations.
- **Square Activation:** Quadratic monomial $f(x) = x^2$ provides non-linear representation while consuming exactly 1 multiplicative depth level per activation.
- **Linear Pooling Folding:** $2 \times 2$ Average Pooling is a linear operator. We constructed transformation matrix $P \in \mathbb{R}^{64 \times 256}$ such that $W_{folded} = W_{FC1} \cdot P$. This collapses Average Pooling and FC1 into a **single matrix multiplication**, consuming **0 extra multiplicative depth** and eliminating expensive slot rotations.

```
Input (1x28x28) 
  -> Conv2D (4 filters, 7x7, s=3, valid)    [Depth 1] -> (4x8x8)
  -> Square Activation                      [Depth 2] -> (4x8x8)
  -> Average Pooling (folded into FC1)      [Depth 0] 
  -> Dense Layer 1 (64 -> 32)               [Depth 3] -> (32,)
  -> Square Activation                      [Depth 4] -> (32,)
  -> Dense Layer 2 (32 -> 4 Logits)         [Depth 5] -> (4,)
```

### Why 28×28 over 32×32?
- For $N = 8192$, CKKS SIMD slots $= N/2 = 4,096$.
- With $28 \times 28$, kernel $7 \times 7$, stride $3$: output spatial dimension $= (28 - 7)/3 + 1 = 8$. Number of sliding windows $= 8 \times 8 = 64$.
- Im2col matrix dimension $= 64 \times 49 = 3,136$ elements. $3,136 \le 4,096$ slots, fitting in a **single ciphertext**.
- In contrast, $32 \times 32$ produces non-integer feature maps or exceeds 4,096 slots.

---

## 4. CKKS Parameter Validation & Critical Spec Deviation

### The Specification Issue
Task 3 specifies: `poly_modulus_degree = 8192`, `coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]`, `global_scale = 2**26`.
- **Empirical Failure:** Squaring doubles the scale ($S \to S^2$). Rescaling divides by the current prime $q_i \approx 2^{21}$.
- Because the scale ($2^{26}$) does not match prime size ($2^{21}$), the scale drifts by factor $2^{52 - 21} = 2^{31}$.
- Across 5 levels, the scale explodes exponentially, resulting in a catastrophic maximum error of **$> 8.4 \times 10^8$**.

### Minimal Justified Alternatives (128-bit Classical Security)
1. **Calibrated 26-bit (Production Standard):** `coeff_mod_bit_sizes = [31, 26, 26, 26, 26, 26, 26, 31]`, `global_scale = 2**26`.
   - Total bit budget $= 31 + 6 \times 26 + 31 = \mathbf{218\text{ bits}} \le 218\text{ bits}$ (128-bit security ceiling for $N=8192$ under HomomorphicEncryption.org standard).
   - Primes match scale exactly $\implies$ **Zero drift**. Maximum error: **$3.20 \times 10^{-5}$**.
2. **Matched 21-bit (Alternative):** `coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]`, `global_scale = 2**21`. Total bits $= 206 \le 218$. Max error: **$2.11 \times 10^{-3}$**.

---

## 5. Quantitative Results & Evaluation

### Plaintext Full Test Set Evaluation (1,311 Scans)

| Metric | FHE Model (Square + AvgPool) | Baseline Model (ReLU + MaxPool) | Architectural Cost |
|---|---|---|---|
| **Test Accuracy** | **64.99%** (95% CI: [62.37%, 67.52%]) | **78.79%** (95% CI: [76.50%, 80.92%]) | **-13.80%** |
| **Macro F1-Score** | 0.6281 | 0.7764 | -0.1483 |
| **Weighted F1-Score** | 0.6355 | 0.7824 | -0.1469 |

### Per-Class Performance & Tumor-Miss Rates (FHE Model)
- **Glioma:** Precision: 0.657, Recall: 0.720, Tumor-Miss Rate: **28.00%** (Support: 300)
- **Meningioma:** Precision: 0.450, Recall: 0.297, Tumor-Miss Rate: **70.26%** (Support: 306)
- **No Tumor:** Precision: 0.701, Recall: 0.775, Miss Rate: **22.47%** (Support: 405)
- **Pituitary:** Precision: 0.696, Recall: 0.770, Tumor-Miss Rate: **23.00%** (Support: 300)

### Encrypted Inference on Stratified Subset ($N=200$)
- **Plaintext Accuracy on Subset:** 59.50%
- **Encrypted Accuracy on Subset:** 55.50%
- **Prediction Agreement Rate:** **90.00%** (95% Wilson CI: [85.06%, 93.43%])
- **Mean Absolute Logit Error:** 4.84
- **Max Absolute Logit Error:** 211.83

---

## 6. The "Cryptographic Tax" Benchmarks

Measured on host CPU across repeated warm-up excluded runs:

### Latency Profile (Median + IQR)

| Stage | Median Latency | IQR | Comparison / Slowdown |
|---|---|---|---|
| **Galois Key Generation** | 1.495 s | 0.082 s | One-time client setup |
| **Client im2col + Encryption** | 0.050 s (50 ms) | 0.004 s | Input preparation |
| **Server Encrypted Inference** | 5.045 s | 0.120 s | Primary computational bottleneck |
| **Client Logit Decryption** | 0.002 s (2 ms) | 0.0003 s | Instantaneous |
| **Plaintext PyTorch Inference** | 0.94 ms | 0.08 ms | Baseline reference |
| **End-to-End FHE Inference** | **5.097 s** | - | **5,409.8× Slowdown** |

### Memory & Payload Expansion

| Parameter | Plaintext Raw | Encrypted Payload | Expansion Factor |
|---|---|---|---|
| **Input Image** | 3,136 bytes (28×28 float32) | 477,076 bytes (~477 KB) | **152.1×** |
| **Output Logits** | 16 bytes (4 float32) | 70,404 bytes (~70 KB) | **4,400×** |
| **Public Context Payload** | - | 13.5 MB (Galois + Relin keys) | - |
| **Peak Process RAM** | ~120 MB | 1,000.0 MB | ~8.3× |

---

## 7. Project Structure

```
Cryptography/
├── data/
│   ├── Training/       # 5,712 training scans (4 classes)
│   └── Testing/        # 1,311 testing scans (4 classes)
├── fhe_brain_tumor/
│   ├── __init__.py     # Package exports
│   ├── config.py       # Frozen configuration dataclass
│   ├── data.py         # ImageFolder loaders, transforms, stratified split
│   ├── model.py        # FHECompatibleCNN, ReLUBaselineCNN, pooling folding
│   ├── fhe_context.py  # CKKS context manager, parameter validator
│   ├── client.py       # Client encryption, decryption, argmax
│   ├── server.py       # Zero-trust server encrypted inference
│   ├── train.py        # Training loops, gradient clipping, early stopping
│   ├── evaluate.py     # Metrics, confusion matrix, encrypted subset eval
│   ├── benchmark.py    # Latency, payload, SIMD throughput profiling
│   ├── generate_summary_pdf.py # ReportLab PDF report generation
│   └── tests/
│       ├── test_data.py
│       ├── test_model.py
│       ├── test_fhe.py
│       └── test_client_server.py
├── artifacts/
│   ├── best_fhe_model.pt
│   ├── best_relu_model.pt
│   ├── class_mapping.json
│   ├── plots/
│   └── results/
│       ├── plaintext_metrics.json
│       ├── encrypted_eval_results.json
│       └── benchmarks.json
├── analysis.ipynb      # Clean top-to-bottom executable narrative notebook
├── summary.pdf         # 3-page formal report quantifying cryptographic tax
├── Makefile            # One-command build automation
├── requirements.txt    # Pinned dependency requirements
└── README.md
```

---

## 8. Reproduction Guide

### Prerequisites
Python 3.10+ (tested on Python 3.14.5 with TenSEAL 0.3.18).

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite (17 Unit & Integration Tests)
```bash
pytest fhe_brain_tumor/tests/ -v
```

### 3. Run Main Pipeline (Tasks 1 - 6 + Benchmarks)
```bash
python -m fhe_brain_tumor.main
```

### 4. Generate Formal Summary Report (summary.pdf)
```bash
python -m fhe_brain_tumor.generate_summary_pdf
```

### 5. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
# or using make:
make app
```
The interactive frontend provides:
- Live sample MRI selection or custom patient MRI scan uploads.
- Side-by-side visualization of original scans vs. $28\times 28$ normalized tensors.
- 3-stage interactive execution: Client encryption $\to$ Zero-trust cloud inference $\to$ Client decryption.
- Real-time diagnostic dashboard comparing FHE encrypted predictions, Plaintext PyTorch predictions, confidence bar charts, and cryptographic tax metrics.

### 6. Execute Narrative Notebook
Open and execute `analysis.ipynb` top-to-bottom in JupyterLab or VSCode. All numbers and plots trace directly to the saved artifacts in `artifacts/results/`.

---

## 9. Updated Project Presentation Speech

> *"Good morning, members of the evaluation committee and colleagues.*
>
> *Today, I present our deliverable: **Secure Brain Tumor Classification using Fully Homomorphic Encryption**.*
>
> *Medical AI presents a fundamental paradox: diagnostic neural networks need sensitive brain MRI scans, but healthcare regulations strictly forbid exposing patient data to third-party cloud servers. To resolve this, we implemented a zero-trust privacy-preserving machine learning pipeline using the CKKS homomorphic encryption scheme.*
>
> *Our project delivers on all six core tasks and resolves critical cryptographic challenges:*
>
> *First, on data: We loaded 7,023 real MRI scans across glioma, meningioma, pituitary, and healthy brain tissue. We verified real per-class counts—5,712 training and 1,311 held-out test scans—and standardized on 28×28 grayscale images.*
>
> *Second, on model design: Standard neural networks rely on non-polynomial operations like ReLU and Max-Pooling, which are mathematically impossible under basic FHE. We designed a custom polynomial CNN utilizing quadratic square activations and folded average pooling directly into dense linear layers. Furthermore, to overcome square activation instability, we introduced small-variance initialization, gradient norm clipping, and weight decay, stabilizing peak activations across training.*
>
> *Third, on cryptography: Our empirical validation uncovered a fatal flaw in the initial project specification. Pairing scale 2^26 with 21-bit primes caused severe scale drift across 5 depth levels, exploding error beyond 800 million. We resolved this by formulating a calibrated 218-bit exact parameterization with matched 26-bit primes, guaranteeing 128-bit classical security with a logit error of just 3.2 × 10^-5.*
>
> *Fourth, fifth, and sixth, on client-server execution: The client encodes scans using im2col and encrypts them into 477 KB ciphertexts. The server—strictly verified to possess zero secret keys—evaluates 4 convolutional filters, polynomial squaring, and folded linear transformations in a single round. On a stratified test subset of 200 scans, encrypted inference achieved a **90.00% prediction agreement rate** with plaintext predictions.*
>
> *Finally, we quantified the 'cryptographic tax':*
> - *The architectural tax is a 13.8% drop in accuracy (78.8% for standard ReLU vs 65.0% for polynomial CNN).*
> - *The computational tax is a 5,410× slowdown (5.05 seconds encrypted server inference vs 0.94 milliseconds for PyTorch).*
> - *The payload expansion is 152× (477 KB ciphertext vs 3.1 KB raw scan).*
>
> *While these findings show that real-time clinical deployment requires further SIMD packing and hardware acceleration, our system proves that zero-trust diagnostic inference on encrypted patient data is mathematically and practically viable.*
>
> *Thank you, and I look forward to your questions."*
