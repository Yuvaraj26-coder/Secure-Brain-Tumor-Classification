"""Client-side preprocessing, encryption, decryption, and classification module."""
import os
import logging
from typing import Tuple, List, Union
import numpy as np
import torch
from PIL import Image
import tenseal as ts
from .config import CONFIG
from .fhe_context import get_fhe_context, get_public_context
from .data import get_base_transform

logger = logging.getLogger(__name__)

class Client:
    """
    Client holding the secret key.
    Preprocesses images, encodes via im2col, encrypts into CKKS ciphertext,
    and decrypts final predictions without disclosing raw MRI scans to the server.
    """
    
    def __init__(self, context: ts.Context = None):
        # Client holds full context (including private key)
        self.context = context if context is not None else get_fhe_context()
        self.public_context = get_public_context()
        self.transform = get_base_transform()
        
    def preprocess_image(self, image_source: Union[str, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess input image to normalized tensor (1, 28, 28) in [0.0, 1.0].
        """
        if isinstance(image_source, str):
            if not os.path.isfile(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            img = Image.open(image_source)
            tensor = self.transform(img)
        elif isinstance(image_source, Image.Image):
            tensor = self.transform(image_source)
        elif isinstance(image_source, torch.Tensor):
            if image_source.ndim == 4:
                tensor = image_source[0]
            else:
                tensor = image_source
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-2:] != (CONFIG.image_size, CONFIG.image_size):
                resize = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(CONFIG.image_size, CONFIG.image_size),
                    mode='bilinear',
                    align_corners=False
                )
                tensor = resize.squeeze(0)
        else:
            raise TypeError(f"Unsupported image type: {type(image_source)}")
            
        return tensor

    def encrypt_image(
        self,
        image_source: Union[str, Image.Image, torch.Tensor]
    ) -> Tuple[bytes, int, int]:
        """
        TWEAK 2: Multi-Window SIMD Packing
        Packs multiple convolution windows into a single TenSEAL CKKS vector 
        utilizing the 2048 available slots from the 4096 degree polynomial.
        """
        import torch.nn.functional as F
        
        tensor = self.preprocess_image(image_source)
        
        # Targeting the new Tweak 4 shallow model: kernel=3, stride=2
        kernel_size = 3
        stride = 2
        
        # Flatten sliding windows using PyTorch's unfold (im2col)
        # tensor shape is (1, 28, 28). unfold requires (1, 1, 28, 28)
        windows = F.unfold(tensor.unsqueeze(0), kernel_size=kernel_size, stride=stride)
        windows = windows.transpose(1, 2).squeeze(0) # Shape: (Num_Windows, Kernel_Elements)
        windows_nb = windows.shape[0]
        
        # Pack into a flat 1D list
        packed_plaintext = []
        for i in range(windows_nb):
            packed_plaintext.extend(windows[i].tolist())
            
        # A poly_modulus_degree of 4096 provides exactly 2048 slots
        slot_count = 2048
        if len(packed_plaintext) < slot_count:
            packed_plaintext += [0.0] * (slot_count - len(packed_plaintext))
        elif len(packed_plaintext) > slot_count:
            raise ValueError("Too many windows to pack into a single 4096-degree ciphertext!")
            
        # Encrypt all windows in one single O(N log N) operation using public context
        enc_vec = ts.ckks_vector(self.public_context, packed_plaintext)
        
        serialized_bytes = enc_vec.serialize()
        payload_size = len(serialized_bytes)
        logger.debug("Encrypted image (packed): windows=%d, size=%d bytes", windows_nb, payload_size)
        
        return serialized_bytes, windows_nb, payload_size

    def encrypt_vector(self, vector: Union[List[float], np.ndarray, torch.Tensor]) -> ts.CKKSVector:
        """Encrypt an arbitrary 1D vector (for tests / benchmarks)."""
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        if isinstance(vector, np.ndarray):
            vec_list = vector.flatten().tolist()
        else:
            vec_list = list(vector)
            
        return ts.ckks_vector(self.public_context, vec_list)

    def decrypt_prediction(self, encrypted_logits: Union[bytes, ts.CKKSVector]) -> Tuple[int, List[float]]:
        """
        Decrypt the returned ciphertext logits using the client's exclusive secret key.
        Computes argmax to obtain predicted class index.
        """
        if isinstance(encrypted_logits, (bytes, bytearray)):
            # Deserialize using client context (which possesses secret key)
            enc_vec = ts.ckks_vector_from(self.context, encrypted_logits)
        else:
            enc_vec = encrypted_logits
            enc_vec.link_context(self.context)
            
        # Decrypt logits
        decrypted_vals = enc_vec.decrypt()
        # Top 4 correspond to the 4 class logits
        logits = decrypted_vals[:CONFIG.num_classes]
        predicted_idx = int(np.argmax(logits))
        return predicted_idx, logits
