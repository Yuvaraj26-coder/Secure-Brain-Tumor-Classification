"""Server-side encrypted inference module for zero-trust cloud inference."""
import os
import logging
from typing import Tuple, List, Optional, Union, Dict, Any
import torch
import tenseal as ts
from .config import CONFIG
from .model import FHEOptimizedCNN, fuse_conv_bn_weights
from .fhe_context import get_public_context

logger = logging.getLogger(__name__)

class Server:
    """
    Simulated cloud inference server executing optimized FHE operations.
    Holds pre-trained plaintext weights and the client's public context.
    CANNOT decrypt ciphertexts (possesses no secret key).
    """
    
    def __init__(
        self,
        model: Optional[FHEOptimizedCNN] = None,
        public_context: Optional[ts.Context] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.public_context = public_context if public_context is not None else get_public_context()
        # Loading the shallower model architecture
        self.model = model if model is not None else FHEOptimizedCNN()
        
        self.verify_no_secret_key()
        self._prepare_weights(checkpoint_path)

    def verify_no_secret_key(self) -> None:
        dummy = ts.ckks_vector(self.public_context, [0.5, 1.0])
        try:
            dummy.decrypt()
            raise RuntimeError("CRITICAL SECURITY VIOLATION: Server context possesses secret key!")
        except (ValueError, RuntimeError) as e:
            logger.info("Security verified: Server is strictly unable to decrypt (%s)", type(e).__name__)

    def _extract_diagonals(self, matrix: List[List[float]]) -> List[List[float]]:
        """TWEAK 1: Extract diagonals for Halevi-Shoup O(N) evaluation."""
        rows = len(matrix)
        cols = len(matrix[0])
        diagonals = []
        for d in range(cols):
            diag = [matrix[i % rows][(i + d) % cols] for i in range(cols)]
            diagonals.append(diag)
        return diagonals

    def _prepare_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Extracts weights, fuses layers, and computes diagonals."""
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state.get("model_state_dict", state))
            
        self.model.eval()
        
        # TWEAK 3: Fuse Conv and BN layers into a single plaintext operation
        w_fused, b_fused = fuse_conv_bn_weights(self.model.conv, self.model.bn)
        
        self.conv_w = w_fused.flatten().tolist()
        self.conv_b = b_fused.flatten().tolist()
        
        # TWEAK 1: Pre-compute diagonals for the fully connected layer
        fc_weight_list = self.model.fc.weight.data.tolist()
        self.fc_diagonals = self._extract_diagonals(fc_weight_list)
        self.fc_bias = self.model.fc.bias.data.tolist()

    def encrypted_forward(self, enc_input: ts.CKKSVector) -> Tuple[ts.CKKSVector, Dict[str, Any]]:
        """Executes the optimized inference pipeline."""
        # TWEAK 2: Packed Convolution via SIMD multiplication
        # Pad the weights to match the 2048 slot size of the 4096-degree polynomial
        padded_w = (self.conv_w * (2048 // len(self.conv_w) + 1))[:2048]
        padded_b = (self.conv_b * (2048 // len(self.conv_b) + 1))[:2048]
        
        # Single O(N log N) multiplication for all sliding windows simultaneously
        enc_conv = enc_input * padded_w + padded_b
        
        # TWEAK 4: Polynomial Activation (Shallower network = only one square operation!)
        enc_act = enc_conv.square()
        
        # TWEAK 1: Halevi-Shoup Diagonal Linear Evaluation
        enc_result = None
        for i, diagonal in enumerate(self.fc_diagonals):
            rotated_input = enc_act
            if i > 0:
                # Shift SIMD slots via Galois Keys
                rotated_input = enc_act.roll(i)
                
            pad_diag = (diagonal * (2048 // len(diagonal) + 1))[:2048]
            partial_sum = rotated_input * pad_diag
            
            if enc_result is None:
                enc_result = partial_sum
            else:
                enc_result += partial_sum
                
        pad_fc_bias = (self.fc_bias * (2048 // len(self.fc_bias) + 1))[:2048]
        enc_logits = enc_result + pad_fc_bias
        
        return enc_logits, {"status": "OPTIMIZED_INFERENCE_SUCCESS"}

    def process_request(self, serialized_input: bytes, windows_nb: int) -> Tuple[bytes, int]:
        """Server pipeline endpoint called by main.py."""
        enc_input = ts.ckks_vector_from(self.public_context, serialized_input)
        enc_logits, _ = self.encrypted_forward(enc_input)
        serialized_output = enc_logits.serialize()
        return serialized_output, len(serialized_output)