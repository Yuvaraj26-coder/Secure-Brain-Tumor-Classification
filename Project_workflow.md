# Project Title: Secure Brain Tumor Classification Using Fully Homomorphic Encryption

**Assigned Date:** \[Insert Date\]\
**Submission Date:** \[Insert Date\]

## Problem Statement

Artificial intelligence systems increasingly rely on highly sensitive
datasets, such as biometric records and medical imaging, making data
privacy paramount. Fully Homomorphic Encryption (FHE) permits arbitrary
computations directly on encrypted data without ever exposing the
underlying plaintext. Your task is to build a privacy-preserving machine
learning (PPML) system that classifies brain tumors using a zero-trust
client-server architecture.

## Dataset --- Where to Get It

-   Navigate to Kaggle and download the **"Brain Tumor MRI Dataset."**
-   The dataset contains over 7,000 human brain MRI images categorized
    into four classes:
    -   Glioma
    -   Meningioma
    -   Pituitary tumor
    -   No tumors
-   The data is pre-divided into training (1,400 images per class) and
    testing (400 images per class) subsets.

## Tasks to Complete

Complete all the following tasks in a Jupyter Notebook.

### Task 1: Dataset Selection and Preprocessing

-   Load the dataset using PyTorch's `ImageFolder` and `DataLoader`.
-   Resize all images to **28 × 28** or **32 × 32** pixels for
    computational feasibility under FHE constraints.
-   Convert the images to grayscale and normalize them to a `[0, 1]`
    range.

### Task 2: Baseline Model Architecture (Plaintext)

-   Adapt a standard Convolutional Neural Network (CNN) to accommodate
    FHE mathematical limitations.
-   Replace standard ReLU activations with a square function,
    `f(x) = x²`, to eliminate non-polynomial operations.
-   Replace Max Pooling with Average Pooling to avoid exact comparative
    operations.
-   Train the model on the plaintext MRI training set using an Adam
    optimizer and `CrossEntropyLoss` until validation accuracy plateaus.

### Task 3: Cryptographic Context Configuration (TenSEAL)

-   Instantiate the FHE context using the TenSEAL library utilizing the
    CKKS scheme.
-   Initialize the context with a polynomial modulus degree of `8192`
    and a coefficient modulus bit-sizes chain of
    `[40, 21, 21, 21, 21, 21, 21, 40]`.
-   Set the global scale to `2` to balance decimal precision and the
    available noise budget.
-   Generate the secret key, public key, and the Galois keys required
    for ciphertext slot rotations.

### Task 4: Client-Side Encryption

-   Extract a normalized MRI scan from the test set representing the
    client-side input.
-   Utilize the **im2col** encoding technique to flatten image windows
    into vectors for homomorphic 2D convolutions.
-   Encrypt the vectors into a ciphertext using the public key and
    transmit them to the simulated cloud server.

### Task 5: Encrypted Cloud Inference

-   Load the pre-trained, unencrypted PyTorch model weights on the
    server side.
-   Process convolutions as highly parallelized matrix multiplications
    over the SIMD slots entirely in the encrypted domain.
-   Apply the square activation by squaring the encrypted vectors
    in-place.
-   Compute linear classification layers via homomorphic dot products
    and bias additions.

### Task 6: Decryption and Final Evaluation

-   Return the final encrypted prediction tensor (the logits) from the
    server to the client.
-   Decrypt the logits using the exclusive secret key.
-   Determine the final tumor classification (0 through 3) by passing
    the resulting floating-point array through a `torch.max` function.
-   Log the comparison between the plaintext model's baseline accuracy
    and the encrypted model's final accuracy, alongside the
    computational latency incurred by the FHE mathematical operations.

## Tools & Libraries to Use

-   **Python 3.x:** Main programming language.
-   **PyTorch:** Framework for training the baseline plaintext model and
    loading image data.
-   **TenSEAL:** Python library optimized for tensor operations on
    encrypted data utilizing the Microsoft SEAL backend.

## What to Submit

-   `analysis.ipynb`: Your complete Jupyter Notebook detailing all 6
    tasks.
-   `summary.pdf`: A comprehensive project report that quantifies the
    **"cryptographic tax."** The report must systematically benchmark
    and discuss:
    -   Model accuracy degradation
    -   Execution latency
    -   Memory payload overhead (ciphertext expansion)

## Important Note for the Developer

Evaluating the efficacy of a PPML architecture necessitates deeply
quantifying the inevitable performance degradation introduced by FHE.
FHE introduces fidelity loss due to random error injected for security
and deviations caused by polynomial approximations. You must clearly
outline the balance between execution throughput via SIMD packing and
computational latency.

A standout project will effectively modularize these paradigms to
demonstrate clean software engineering practices alongside cryptographic
implementation.
