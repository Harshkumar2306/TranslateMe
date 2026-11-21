"""
Residual Connections and Layer Normalization
Section 3.1 & 5.4 of 'Attention Is All You Need'
"""

import torch
import torch.nn as nn
from typing import Callable


class LayerNorm(nn.Module):
    """
    Standard Layer Normalization layer:
    LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer normalization.
    Supports standard Post-LN: LayerNorm(x + Dropout(Sublayer(x)))
    and Pre-LN: x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, size: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.pre_ln = pre_ln

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Apply residual connection to any sublayer with the same size.
        """
        if self.pre_ln:
            # Pre-LN: Stable gradient flow in deep transformers
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            # Post-LN: Original Vaswani et al. architecture
            return self.norm(x + self.dropout(sublayer(x)))
