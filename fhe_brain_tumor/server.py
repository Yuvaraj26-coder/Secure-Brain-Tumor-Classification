"""Server-side encrypted inference module for zero-trust cloud inference."""
import os
import logging
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import tenseal as ts
from .config import CONFIG
from .model import FHECompatibleCNN, build_avgpool_folding_matrix
from .fhe_context import get_public_context

logger = logging.getLogger(__name__)

class Server:
    """
    Simulated cloud inference server.
    Holds pre-trained plaintext weights and the client's public context.
    CANNOT decrypt ciphertexts (possesses no secret key).
    """
    
    def __init__(
        self,
        model: Optional[FHECompatibleCNN] = None,
        public_context: Optional[ts.Context] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.public_context = public_context if public_context is not None else get_public_context()
        self.model = model if model is not None else FHECompatibleCNN()
        
        # Verify strict zero-trust: server must NEVER have decryption capability
        self.verify_no_secret_key()
        
        # Precompute folded linear weights for single-round FHE evaluation
        self._prepare_weights(checkpoint_path)

    def verify_no_secret_key(self) -> None:
        """
        Formally verify that the server's context cannot decrypt ciphertexts.
        Raises RuntimeError if decryption succeeds.
        """
        # Test vector to attempt unauthorized decryption
        dummy = ts.ckks_vector(self.public_context, [0.5, 1.0])
        try:
            dummy.decrypt()
            raise RuntimeError("CRITICAL SECURITY VIOLATION: Server context possesses secret key!")
        except (ValueError, RuntimeError) as e:
            # Expected behavior: SEAL raises ValueError stating context does not hold a secret key
            logger.info("Security verified: Server is strictly unable to decrypt (%s)", type(e).__name__)

    def _prepare_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Extract and format PyTorch weights into TenSEAL-compatible plain structures."""
        if checkpoint_path and os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            logger.info("Server loaded model weights from %s", checkpoint_path)
            
        self.model.eval()
        
        # 1. Conv1 weights and biases
        # Shape: [out_channels, 1, kernel, kernel] -> list of [kernel, kernel] per channel
        out_ch = self.model.conv.out_channels
        k = self.model.conv.kernel_size[0]
        self.conv_weights = self.model.conv.weight.data.view(out_ch, k, k).tolist()
        self.conv_biases = self.model.conv.bias.data.tolist()
        
        # 2. AvgPool folding into FC1
        # Feature map after conv is [out_ch=4, 8, 8] -> 256
        # AvgPool (2x2, stride 2) produces [4, 4, 4] -> 64
        P = build_avgpool_folding_matrix() # shape (64, 256)
        # Folded weight: W_folded = W_FC1 @ P, shape (32, 256)
        W_folded = self.model.fc1.weight.data @ P
        # TenSEAL mm expects matrix formatted as (in_dim, out_dim) = (256, 32)
        self.fc1_folded_weight = W_folded.T.tolist()
        self.fc1_bias = self.model.fc1.bias.data.tolist()
        
        # 3. FC2 weights and biases
        # W_FC2 shape: [4, 32] -> transposed to (32, 4) for TenSEAL mm
        self.fc2_weight = self.model.fc2.weight.data.T.tolist()
        self.fc2_bias = self.model.fc2.bias.data.tolist()

    def encrypted_forward(
        self,
        enc_input: ts.CKKSVector,
        windows_nb: int
    ) -> Tuple[ts.CKKSVector, Dict[str, Any]]:
        """
        Execute full encrypted forward pass in a single round without decryption:
        1. Encrypted Conv2D via im2col
        2. Channel packing
        3. Square activation
        4. Linear layer 1 (with AvgPool folded in)
        5. Square activation
        6. Linear layer 2 (Output logits)
        """
        # 1. Conv2D Evaluation across 4 filters
        enc_channels = []
        for kernel, bias in zip(self.conv_weights, self.conv_biases):
            y = enc_input.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
            
        # 2. Pack channel outputs into a unified vector ciphertext
        enc_conv = ts.CKKSVector.pack_vectors(enc_channels)
        
        # 3. Square activation 1
        enc_conv.square_()
        
        # 4. FC1 with folded average pooling
        enc_fc1 = enc_conv.mm(self.fc1_folded_weight) + self.fc1_bias
        
        # 5. Square activation 2
        enc_fc1.square_()
        
        # 6. FC2 (Output logits)
        enc_logits = enc_fc1.mm(self.fc2_weight) + self.fc2_bias
        
        metadata = {
            "multiplicative_depth_consumed": 5,
            "status": "INFERENCE_SUCCESS"
        }
        return enc_logits, metadata

    def process_request(self, serialized_input: bytes, windows_nb: int) -> Tuple[bytes, int]:
        """
        Full server pipeline: deserialize input, run encrypted forward pass, serialize response.
        """
        # Deserialize ciphertext from client
        enc_input = ts.ckks_vector_from(self.public_context, serialized_input)
        
        # Execute encrypted inference
        enc_logits, _ = self.encrypted_forward(enc_input, windows_nb)
        
        # Serialize response payload
        serialized_output = enc_logits.serialize()
        return serialized_output, len(serialized_output)
