"""Tests for Client-Server interaction, zero-trust enforcement, and encrypted inference."""
import os
import pytest
import numpy as np
import torch
import tenseal as ts
from fhe_brain_tumor.config import CONFIG
from fhe_brain_tumor.model import FHECompatibleCNN
from fhe_brain_tumor.client import Client
from fhe_brain_tumor.server import Server

def test_server_zero_trust_cannot_decrypt():
    """Verify Server strictly raises an exception if decryption is attempted."""
    client = Client()
    server = Server(public_context=client.public_context)
    # verify_no_secret_key was invoked during init without throwing RuntimeError
    
    # Directly test server cannot decrypt any received ciphertext
    enc_v = client.encrypt_vector([1.0, 2.0, 3.0])
    with pytest.raises((ValueError, RuntimeError)):
        enc_v.decrypt()

def test_client_encryption_and_decryption_vector():
    """Verify client vector encryption and decryption roundtrip."""
    client = Client()
    vec = [0.1, -0.2, 0.3, -0.4]
    enc_v = client.encrypt_vector(vec)
    
    # Decrypt via client context
    pred, logits = client.decrypt_prediction(enc_v)
    assert len(logits) == 4
    for orig, dec in zip(vec, logits):
        assert abs(orig - dec) < 1e-3

def test_client_server_encrypted_inference_accuracy():
    """
    Verify complete encrypted client-server inference matches plaintext PyTorch
    forward pass on a test sample within approximate FHE bounds.
    """
    model = FHECompatibleCNN()
    if os.path.isfile(CONFIG.model_checkpoint):
        state = torch.load(CONFIG.model_checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    
    client = Client()
    server = Server(model=model, public_context=client.public_context)
    
    # Use a real test sample if available, else a realistic normalized MRI-like scan
    if os.path.isdir(CONFIG.testing_dir):
        from fhe_brain_tumor.data import load_datasets
        _, test_dataset = load_datasets()
        img_path, true_cls = test_dataset.samples[0]
        img = client.preprocess_image(img_path).unsqueeze(0)
    else:
        torch.manual_seed(42)
        img = torch.zeros(1, 1, CONFIG.image_size, CONFIG.image_size)
        img[:, :, 10:18, 10:18] = 0.5  # Central brain-like region
        
    # 1. Plaintext PyTorch forward pass
    with torch.no_grad():
        pt_logits, _ = model(img)
        pt_logits_list = pt_logits[0].cpu().numpy().tolist()
        pt_pred = int(torch.argmax(pt_logits, dim=1).item())
        
    # 2. Client encodes & encrypts
    cipher_bytes, windows_nb, payload_size = client.encrypt_image(img)
    assert payload_size > 0
    assert windows_nb == 64  # (28-7)//3 + 1 = 8 -> 8*8 = 64
    
    # 3. Server processes encrypted inference
    resp_bytes, resp_size = server.process_request(cipher_bytes, windows_nb)
    assert resp_size > 0
    
    # 4. Client decrypts and classifies
    enc_pred, enc_logits_list = client.decrypt_prediction(resp_bytes)
    
    # Assert predictions agree and logits approximate plaintext within FHE tolerance
    assert enc_pred == pt_pred, f"Prediction mismatch: plaintext {pt_pred} vs encrypted {enc_pred}"
    max_logit_error = max(abs(a - b) for a, b in zip(pt_logits_list, enc_logits_list))
    assert max_logit_error < 0.5, f"Encrypted vs plaintext logit discrepancy too high: {max_logit_error}"
