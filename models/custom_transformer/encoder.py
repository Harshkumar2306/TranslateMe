"""
Transformer Encoder and EncoderLayer
Section 3.1 of 'Attention Is All You Need'
"""

import copy
from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .sublayers import SublayerConnection, LayerNorm
from .embeddings import TransformerEmbedding


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderLayer(nn.Module):
    """
    Encoder is composed of a stack of N = 6 (or configurable) identical layers.
    Each layer has two sub-layers:
      1. Multi-head self-attention mechanism
      2. Position-wise fully connected feed-forward network
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln)
        ])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional source padding mask (batch_size, 1, 1, seq_len)
        Returns:
            Encoded representation (batch_size, seq_len, d_model)
        """
        # Sublayer 1: Multi-Head Self-Attention
        x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        # Sublayer 2: Feed-Forward Network
        x = self.sublayer_connections[1](x, self.feed_forward)
        return x


class TransformerEncoder(nn.Module):
    """
    Core Transformer Encoder stack.
    Passes input through Token+Positional embeddings, then through N stacked EncoderLayers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 512,
        max_len: int = 5000,
        dropout: float = 0.1,
        pre_ln: bool = False
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len=max_len, dropout=dropout)
        self.layers = clones(EncoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Source token IDs (batch_size, seq_len)
            mask: Source padding mask (batch_size, 1, 1, seq_len)
        Returns:
            Contextual memory representations (batch_size, seq_len, d_model)
        """
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
