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
        Encode image using ts.im2col_encoding with model's kernel (7x7) and stride (3).
        Returns:
          serialized_ciphertext: bytes representing the encrypted image payload
          windows_nb: number of sliding windows (64 for 28x28 with k=7, s=3)
          payload_size_bytes: size in bytes of the serialized ciphertext
        """
        tensor = self.preprocess_image(image_source)
        # 2D grayscale image as list of lists
        image_2d = tensor[0].tolist()
        
        # TenSEAL im2col_encoding
        # Note: encrypt using public context to simulate true client encryption
        enc_vec, windows_nb = ts.im2col_encoding(
            self.public_context,
            image_2d,
            kernel_n_rows=CONFIG.conv_kernel_size,
            kernel_n_cols=CONFIG.conv_kernel_size,
            stride=CONFIG.conv_stride
        )
        
        serialized_bytes = enc_vec.serialize()
        payload_size = len(serialized_bytes)
        logger.debug("Encrypted image: windows=%d, size=%d bytes", windows_nb, payload_size)
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
