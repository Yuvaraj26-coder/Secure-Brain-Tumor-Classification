# ROLE
You are a senior ML + applied-cryptography engineer. Improve an existing, incomplete
project into a professional-grade, reproducible deliverable. Be rigorous and honest:
report negative results, never fabricate numbers, and run everything you claim.

# CONTEXT
Project: "Secure Brain Tumor Classification using Fully Homomorphic Encryption"
(CKKS via TenSEAL, PyTorch, Kaggle "Brain Tumor MRI Dataset": glioma, meningioma,
notumor, pituitary). Zero-trust client–server inference.

Current state: only a plaintext training notebook exists (read it first). It has these
defects you must fix:
1. Only part of Task 2 is done. Tasks 1, 3, 4, 5, 6 and summary.pdf are missing.
2. Images are 64×64 (spec: 28×28 or 32×32); normalization is ~[-1,1] (spec: [0,1]).
3. No average pooling (spec: avg-pool replaces max-pool).
4. No validation split, no early stopping, test set never used; the sanity check uses a
   training image; only training loss is reported.
5. No metrics (accuracy/precision/recall/F1/confusion matrix), no seeds, no checkpointing.
6. No handling of square-activation instability.
7. Not modular; config lives in .env; no tests or README.

# DELIVERABLES
- analysis.ipynb: clean narrative notebook covering all 6 tasks, importing from modules.
- summary.pdf: report quantifying the "cryptographic tax".
- Python package, README, pinned requirements.txt, tests, saved artifacts.

Suggested layout:

fhe_brain_tumor/
  config.py          # frozen dataclass of hyperparameters, FHE params, paths, seeds
  data.py            # ImageFolder, stratified train/val split, loaders, class mapping
  model.py           # FHE-friendly CNN (square act., avg pool, no non-polynomial ops)
  train.py           # training loop, early stopping on val acc, best-checkpoint save
  evaluate.py        # accuracy, per-class P/R/F1, confusion matrix, plots
  fhe_context.py     # TenSEAL CKKS context, key generation, serialization helpers
  client.py          # preprocessing, im2col encoding, encryption, decryption, argmax
  server.py          # loads weights, encrypted forward pass (no secret key access)
  benchmark.py       # latency, memory, accuracy-degradation, error metrics
  tests/             # pytest
analysis.ipynb  summary.pdf  README.md  requirements.txt  artifacts/


# TASK REQUIREMENTS

## Task 1 – Data
- Use ImageFolder + DataLoader on Training/ and Testing/. Measure and report the
  REAL per-class counts (do not assume 1,400/400).
- Grayscale, resize to 28×28 or 32×32 (pick one, justify with accuracy vs FHE cost),
  scale to [0,1] via ToTensor only (no mean/std shift unless folded into the model
  and documented).
- Stratified train/validation split from Training/ (e.g. 90/10, fixed seed). The
  Testing/ set is touched ONLY for final evaluation.
- Print and assert the class-to-index mapping; save it with the model artifact.
- Optional light augmentation (small rotation/translation) applied to train only;
  show whether it helps.

## Task 2 – Plaintext FHE-compatible model
- Conv → square → AvgPool → flatten → FC → square → FC (4 logits). No ReLU, no
  max-pool, no batch-norm, no softmax inside the model.
- Choose kernel/stride so im2col dimensions are small enough for 4096 CKKS slots
  (N=8192). Document the conv output shape math and the multiplicative depth.
- Mitigate square-activation instability: small-variance init, gradient clipping,
  weight decay, optional LR scheduler. Monitor max |activation| and report it.
- Adam + CrossEntropyLoss; early stopping on VALIDATION accuracy plateau; save the
  best checkpoint (state_dict + config + class mapping).
- Report on the held-out test set: accuracy, per-class precision/recall/F1,
  confusion matrix (plot). Comment on tumor-miss rate (recall per class).
- Seed everything (torch, numpy, random, DataLoader workers); report mean ± std over
  ≥3 seeds if compute allows.
- Also train a standard ReLU/max-pool baseline of the same size so the cost of the
  FHE-friendly changes is quantified.

## Task 3 – CKKS context
- Create the context exactly as specified: poly_modulus_degree=8192,
  coeff_mod_bit_sizes=[40,21,21,21,21,21,21,40], global_scale=2**26,
  generate_galois_keys(), public/secret keys.
- CRITICAL: validate these parameters empirically. Scale 2^26 with 21-bit primes
  makes the scale drift across rescales, and the total bit budget and the number of
  levels (needed ≈5: conv, square, FC, square, FC) must be checked.
  1. Run the spec parameters on a small test and measure max logit error vs plaintext.
  2. If precision fails or errors blow up, do NOT silently change them. Report the
     failure, then propose and test the minimal justified alternative (for example
     matching scale and prime sizes, staying within the 128-bit-secure bit budget for
     N=8192, which is 218 bits). Keep the spec version as the reference run.
  3. Confirm the 128-bit security level in the documentation of the chosen params.
- Provide serialization helpers: context.serialize(save_secret_key=False) or
  make_context_public() for the server; the secret key never leaves the client.

## Task 4 – Client encryption
- Take normalized MRI test images (client input). Use TenSEAL's im2col encoding
  (ts.im2col_encoding) matching the model's conv kernel/stride.
- Encrypt with the public-side context; serialize the ciphertext (bytes) to simulate
  network transfer. Report ciphertext size in bytes.

## Task 5 – Server encrypted inference
- Server loads plaintext trained weights and receives only the public context and
  ciphertext. Add a test proving the server object cannot decrypt (no secret key).
- Conv via encrypted-matrix × plaintext-weights (conv2d_im2col style), square via
  ciphertext self-multiplication, average pooling folded into linear ops where
  possible, FC via mm/dot with plaintext weights + bias.
- Process in the encrypted domain only; no decrypt/re-encrypt in between (single
  round). Log the multiplicative depth consumed and the remaining levels.

## Task 6 – Decrypt and evaluate
- Client decrypts the logits with its secret key, then applies torch.max/argmax.
- Evaluate encrypted inference on a STRATIFIED random subset of the test set
  (N ≥ 200 if feasible, equal class coverage; justify N versus runtime).
  Report:
  - plaintext accuracy on the same subset vs encrypted accuracy
  - prediction agreement rate (encrypted vs plaintext)
  - logit error: mean/max absolute error, per-class
  - 95% confidence intervals (e.g. Wilson/bootstrap)
- Log everything in a results table (CSV/JSON in artifacts/).

## "Cryptographic tax" benchmarks (feed summary.pdf)
Measure and tabulate, with repeated runs (median + IQR, warm-up excluded):
- Latency per stage: keygen, context setup, im2col+encrypt, server inference,
  decrypt; plaintext PyTorch inference time; slowdown factor.
- Memory/payload: serialized sizes of public context, Galois keys, ciphertext input,
  ciphertext output; ciphertext expansion factor vs the raw image bytes; peak RAM.
- Accuracy degradation: plaintext → encrypted (and ReLU baseline → square model).
- Throughput vs latency: investigate SIMD packing (multiple images or windows per
  ciphertext) and explain the trade-off with measured numbers.
- Sensitivity: scale/modulus choices vs error and latency (small ablation).

## Security and threat-model section (in summary.pdf)
State assumptions: honest-but-curious server, client holds the secret key, server
sees only ciphertexts, plaintext model weights are NOT private. State CKKS caveats
honestly: approximate arithmetic, IND-CPA only, no integrity/verifiability, leakage of
input dimensions/timing, key/ciphertext size costs. State that this is a research demo,
NOT clinically validated.

# ENGINEERING STANDARDS
- Type hints, docstrings, no magic numbers (all in config.py), logging not prints.
- pytest tests: data shapes and label mapping, model forward shape, no non-polynomial
  ops in the model (inspect modules), encrypt→decrypt round-trip error < tolerance,
  server cannot hold a secret key, encrypted logits ≈ plaintext logits on a sample.
- Pin versions; verify TenSEAL wheel compatibility with the Python version used and
  document the install steps. Provide a one-command run (Makefile or script).
- Notebook must run top-to-bottom on a clean kernel; keep heavy encrypted evaluation
  behind a config flag with a cached results file.
- No fabricated or hard-coded results: every number in the report must come from a
  saved artifact produced by the code.

# SUMMARY.PDF STRUCTURE
1. Abstract  2. Problem and threat model  3. Data and preprocessing  4. Model design
and FHE constraints (depth analysis)  5. CKKS parameter choice, validation and
deviations  6. Results: plaintext baselines, encrypted accuracy, agreement
7. Cryptographic tax: latency, memory/ciphertext expansion, throughput (tables +
plots)  8. Limitations and risks  9. Conclusion and future work (e.g. smaller
ciphertext via packing, bootstrapping-free deeper nets, validation on external data).

# ALSO
Rewrite the project presentation speech so it matches the finished project: all six
tasks, real measured numbers, the limitations, and the cryptographic-tax findings.

# DEFINITION OF DONE
All six tasks run end-to-end; the test suite passes; all numbers in summary.pdf trace
to artifacts; spec deviations are listed explicitly with evidence; the README lets a
newcomer reproduce everything.
Finish with a short report: what was done, what failed, and what remains uncertain.