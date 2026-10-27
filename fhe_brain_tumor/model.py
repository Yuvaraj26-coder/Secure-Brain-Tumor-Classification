"""FHE-compatible CNN model and baseline architectures."""
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
from .config import CONFIG

class SquareActivation(nn.Module):
    """Square activation function f(x) = x^2, strictly polynomial for CKKS FHE."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(x)

class FHECompatibleCNN(nn.Module):
    """
    FHE-compatible CNN architecture strictly adhering to the specification:
    Input (1x28x28) 
      -> Conv2D (4 kernels, 7x7, stride 3, no padding) -> (4x8x8)
      -> Square Activation -> (4x8x8)
      -> AvgPool2D (2x2, stride 2) -> (4x4x4)
      -> Flatten -> (64,)
      -> FC1 (64 -> 32) -> (32,)
      -> Square Activation -> (32,)
      -> FC2 (32 -> 4) -> (4 logits)
    
    Zero non-polynomial operations: No ReLU, no MaxPool, no BatchNorm, no Softmax.
    Multiplicative depth:
      - Conv2D: 1 depth (encrypted input * plain filter weights)
      - Square 1: 1 depth (encrypted vector * encrypted vector)
      - FC1 (with AvgPool folded in): 1 depth (encrypted vector * plain folded weights)
      - Square 2: 1 depth (encrypted vector * encrypted vector)
      - FC2: 1 depth (encrypted vector * plain weights)
      Total depth = 5 multiplications.
    """
    
    def __init__(
        self,
        in_channels: int = CONFIG.conv_in_channels,
        out_channels: int = CONFIG.conv_out_channels,
        kernel_size: int = CONFIG.conv_kernel_size,
        stride: int = CONFIG.conv_stride,
        hidden_dim: int = CONFIG.fc_hidden_dim,
        num_classes: int = CONFIG.num_classes,
        init_std: float = CONFIG.weight_init_std
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 1. Convolutional Layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True
        )
        
        # 2. Square Activation 1
        self.sq1 = SquareActivation()
        
        # 3. Average Pooling (2x2, stride 2)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=CONFIG.avg_pool_kernel_size,
            stride=CONFIG.avg_pool_stride
        )
        
        # Calculated dimensions
        # (28 - 7)//3 + 1 = 8 spatial dimension
        # After 2x2 pool: 8 // 2 = 4 spatial dimension
        self.flatten_dim = out_channels * CONFIG.pool_output_spatial * CONFIG.pool_output_spatial
        
        # 4. Fully Connected Layer 1
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim, bias=True)
        
        # 5. Square Activation 2
        self.sq2 = SquareActivation()
        
        # 6. Fully Connected Layer 2 (Output Logits)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
        
        # Small-variance initialization to prevent exponential activation blowup with square activations
        self._initialize_weights(init_std)
    
    def _initialize_weights(self, std: float) -> None:
        """Initialize weights with small variance normal distribution to mitigate square instability."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor, track_max_act: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Forward pass.
        Returns logits, and optionally the maximum observed absolute activation across layers.
        """
        max_act = 0.0
        
        # Conv
        x = self.conv(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        # Square 1
        x = self.sq1(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        # AvgPool
        x = self.avg_pool(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC1
        x = self.fc1(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        # Square 2
        x = self.sq2(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        # FC2 (Raw Logits)
        x = self.fc2(x)
        if track_max_act:
            max_act = max(max_act, float(x.detach().abs().max()))
            
        return x, max_act

class ReLUBaselineCNN(nn.Module):
    """
    Standard CNN baseline of equivalent topology using ReLU and MaxPool
    to directly quantify the performance impact ('architectural tax') of FHE constraints.
    """
    
    def __init__(
        self,
        in_channels: int = CONFIG.conv_in_channels,
        out_channels: int = CONFIG.conv_out_channels,
        kernel_size: int = CONFIG.conv_kernel_size,
        stride: int = CONFIG.conv_stride,
        hidden_dim: int = CONFIG.fc_hidden_dim,
        num_classes: int = CONFIG.num_classes
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)
        self.relu1 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=CONFIG.avg_pool_kernel_size, stride=CONFIG.avg_pool_stride)
        
        self.flatten_dim = out_channels * CONFIG.pool_output_spatial * CONFIG.pool_output_spatial
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim, bias=True)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=True)
        
        # Standard Kaiming Normal initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu1(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

def build_avgpool_folding_matrix(
    channels: int = CONFIG.conv_out_channels,
    in_spatial: int = CONFIG.conv_output_spatial,
    pool_size: int = CONFIG.avg_pool_kernel_size,
    pool_stride: int = CONFIG.avg_pool_stride
) -> torch.Tensor:
    """
    Construct linear transformation matrix P that folds 2D Average Pooling into FC1.
    Given flattened conv features x of shape (channels * in_spatial * in_spatial),
    pooled features = P @ x has shape (channels * out_spatial * out_spatial).
    
    This enables computing FC1 directly in the encrypted domain in a single linear step:
    W_folded = W_FC1 @ P, consuming 0 extra multiplicative depth and zero rotations.
    """
    out_spatial = (in_spatial - pool_size) // pool_stride + 1
    in_dim = channels * in_spatial * in_spatial
    out_dim = channels * out_spatial * out_spatial
    
    P = torch.zeros(out_dim, in_dim, dtype=torch.float32)
    weight_val = 1.0 / (pool_size * pool_size)
    
    for c in range(channels):
        for i in range(out_spatial):
            for j in range(out_spatial):
                out_idx = c * (out_spatial * out_spatial) + i * out_spatial + j
                for di in range(pool_size):
                    for dj in range(pool_size):
                        row = i * pool_stride + di
                        col = j * pool_stride + dj
                        in_idx = c * (in_spatial * in_spatial) + row * in_spatial + col
                        P[out_idx, in_idx] = weight_val
    return P

def inspect_model_fhe_compliance(model: nn.Module) -> Dict[str, Any]:
    """Inspect model layers and certify zero non-polynomial operations."""
    non_poly = []
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.BatchNorm2d, nn.Softmax)):
            non_poly.append((name, module.__class__.__name__))
            
    return {
        "is_fhe_compliant": len(non_poly) == 0,
        "non_poly_violations": non_poly,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "multiplicative_depth": 5
    }
