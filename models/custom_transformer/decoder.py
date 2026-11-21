"""
Transformer Decoder and DecoderLayer
Section 3.1 of 'Attention Is All You Need'
"""

from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .sublayers import SublayerConnection, LayerNorm
from .embeddings import TransformerEmbedding
from .encoder import clones


class DecoderLayer(nn.Module):
    """
    Decoder is composed of a stack of N identical layers.
    In addition to the two sub-layers in each encoder layer,
    the decoder inserts a third sub-layer, which performs multi-head attention
    over the output of the encoder stack (Cross-Attention).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.d_model = d_model
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 2. Cross-Attention (Query from decoder, Key/Value from encoder)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 3. Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln)
        ])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Target tokens tensor (batch_size, tgt_len, d_model)
            memory: Encoder representations (batch_size, src_len, d_model)
            src_mask: Source padding mask (batch_size, 1, 1, src_len)
            tgt_mask: Target causal + padding mask (batch_size, 1, tgt_len, tgt_len)
        Returns:
            Decoder representation (batch_size, tgt_len, d_model)
        """
        # Sublayer 1: Masked Self-Attention (prevents leftward information flow)
        x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))

        # Sublayer 2: Cross-Attention (Q from decoder x, K/V from encoder memory)
        x = self.sublayer_connections[1](x, lambda _x: self.cross_attn(_x, memory, memory, src_mask))

        # Sublayer 3: Feed-Forward
        x = self.sublayer_connections[2](x, self.feed_forward)
        return x


class TransformerDecoder(nn.Module):
    """
    Core Transformer Decoder stack.
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
        self.layers = clones(DecoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
