"""
Transformer Encoder and EncoderLayer
Section 3.1 of 'Attention Is All You Need' (Vaswani et al., 2017)
"""

import copy
from typing import Optional, List, Tuple
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
    EncoderLayer consists of two sub-layers:
      1. Multi-head self-attention mechanism
      2. Position-wise fully connected feed-forward network
    with residual connections around each and layer normalization.
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Source padding mask (batch_size, 1, 1, seq_len)
            return_attn: Whether to return self-attention weights
        """
        attn_weights = None
        if return_attn:
            # Manual sublayer pass to capture attention weights
            norm_x = self.sublayer_connections[0].norm(x) if self.sublayer_connections[0].pre_ln else x
            attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask=mask, return_attn=True)
            if self.sublayer_connections[0].pre_ln:
                x = x + self.sublayer_connections[0].dropout(attn_out)
            else:
                x = self.sublayer_connections[0].norm(x + self.sublayer_connections[0].dropout(attn_out))
        else:
            x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))

        # Sublayer 2: Feed-Forward
        x = self.sublayer_connections[1](x, self.feed_forward)
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Core Transformer Encoder stack.
    Passes input through Token+Positional embeddings, then through N stacked EncoderLayers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        pre_ln: bool = False,
        embedding: Optional[TransformerEmbedding] = None
    ):
        super().__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = TransformerEmbedding(vocab_size, d_model, max_len=max_len, dropout=dropout)

        self.layers = clones(EncoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            src: Source token IDs (batch_size, seq_len)
            mask: Source padding mask (batch_size, 1, 1, seq_len)
            return_all_attentions: Return list of self-attention matrices from each layer
        Returns:
            encoded_representation or (encoded_representation, attention_list)
        """
        x = self.embedding(src)
        attentions = []

        for layer in self.layers:
            x, attn = layer(x, mask, return_attn=return_all_attentions)
            if return_all_attentions:
                attentions.append(attn)

        norm_out = self.norm(x)
        if return_all_attentions:
            return norm_out, attentions
        return norm_out
