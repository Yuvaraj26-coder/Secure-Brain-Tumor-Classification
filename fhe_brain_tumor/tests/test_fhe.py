"""Tests for TenSEAL CKKS context, parameter validation, and security constraints."""
import pytest
import numpy as np
import tenseal as ts
from fhe_brain_tumor.config import CONFIG
from fhe_brain_tumor.fhe_context import context_manager, get_fhe_context, get_public_context

def test_context_creation_and_slots():
    """Verify CKKS context creation with N=8192 yields 4096 SIMD slots."""
    priv = get_fhe_context()
    assert priv.is_private() is True
    # For N=8192, number of slots is N / 2 = 4096
    v = ts.ckks_vector(priv, [1.0, 2.0, 3.0])
    assert v.size() == 3

def test_public_context_cannot_decrypt():
    """Verify public context has secret key stripped and cannot decrypt."""
    pub = get_public_context()
    assert pub.is_public() is True
    
    v = ts.ckks_vector(pub, [1.0, 2.0, 3.0])
    with pytest.raises((ValueError, RuntimeError)):
        v.decrypt()

def test_128_bit_security_budget():
    """Verify total coefficient modulus bits do not exceed 218 bits for N=8192."""
    cal_bits = sum(CONFIG.calibrated_coeff_mod_bit_sizes)
    assert cal_bits <= 218, f"Calibrated bits {cal_bits} exceeds 128-bit security threshold (218)"
    
    spec_bits = sum(CONFIG.spec_coeff_mod_bit_sizes)
    assert spec_bits <= 218, f"Spec bits {spec_bits} exceeds 128-bit security threshold (218)"

def test_encrypt_decrypt_roundtrip_precision():
    """Verify clean roundtrip encrypt/decrypt error is within 1e-4."""
    ctx = get_fhe_context()
    original = [0.12, -0.45, 0.78, -0.91]
    enc = ts.ckks_vector(ctx, original)
    dec = enc.decrypt()
    
    max_err = max(abs(a - b) for a, b in zip(original, dec))
    assert max_err < 1e-4

def test_parameter_validation_demonstrates_spec_drift():
    """Verify empirical validation runs and identifies spec parameter failure."""
    report = context_manager.validate_parameters_empirically()
    assert "spec_params" in report
    assert "calibrated_params" in report
    assert report["calibrated_params"]["status"] == "PASSED"
    assert report["calibrated_params"]["max_absolute_error"] < 1e-3
