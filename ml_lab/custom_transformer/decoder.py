"""
Transformer Decoder and DecoderLayer
Section 3.1 of 'Attention Is All You Need' (Vaswani et al., 2017)
"""

from typing import Optional, List, Tuple, Dict
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward
from .sublayers import SublayerConnection, LayerNorm
from .embeddings import TransformerEmbedding
from .encoder import clones


class DecoderLayer(nn.Module):
    """
    DecoderLayer is composed of three sub-layers:
      1. Masked Multi-Head Self-Attention (prevents positions from attending to subsequent positions)
      2. Multi-Head Cross-Attention (Query from decoder, Key/Value from encoder memory)
      3. Position-wise Feed-Forward Network
    with residual connections around each and layer normalization.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.d_model = d_model
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 2. Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        # 3. Position-wise Feed-Forward
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
        tgt_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Target tokens tensor (batch_size, tgt_len, d_model)
            memory: Encoder representations (batch_size, src_len, d_model)
            src_mask: Source padding mask (batch_size, 1, 1, src_len)
            tgt_mask: Target causal + padding mask (batch_size, 1, tgt_len, tgt_len)
            return_attn: Whether to return self and cross attention matrices
        Returns:
            (output, self_attn_weights, cross_attn_weights)
        """
        self_attn_w = None
        cross_attn_w = None

        if return_attn:
            # Sublayer 1: Masked Self-Attention
            norm_x1 = self.sublayer_connections[0].norm(x) if self.sublayer_connections[0].pre_ln else x
            s_out, self_attn_w = self.self_attn(norm_x1, norm_x1, norm_x1, mask=tgt_mask, return_attn=True)
            if self.sublayer_connections[0].pre_ln:
                x = x + self.sublayer_connections[0].dropout(s_out)
            else:
                x = self.sublayer_connections[0].norm(x + self.sublayer_connections[0].dropout(s_out))

            # Sublayer 2: Cross-Attention
            norm_x2 = self.sublayer_connections[1].norm(x) if self.sublayer_connections[1].pre_ln else x
            c_out, cross_attn_w = self.cross_attn(norm_x2, memory, memory, mask=src_mask, return_attn=True)
            if self.sublayer_connections[1].pre_ln:
                x = x + self.sublayer_connections[1].dropout(c_out)
            else:
                x = self.sublayer_connections[1].norm(x + self.sublayer_connections[1].dropout(c_out))
        else:
            # Sublayer 1: Masked Self-Attention
            x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
            # Sublayer 2: Cross-Attention
            x = self.sublayer_connections[1](x, lambda _x: self.cross_attn(_x, memory, memory, src_mask))

        # Sublayer 3: Feed-Forward Network
        x = self.sublayer_connections[2](x, self.feed_forward)
        return x, self_attn_w, cross_attn_w


class TransformerDecoder(nn.Module):
    """
    Core Transformer Decoder stack.
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

        self.layers = clones(DecoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            tgt: Target token IDs (batch_size, tgt_len)
            memory: Contextual representations from encoder (batch_size, src_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            return_all_attentions: Returns cross-attention weights per layer
        Returns:
            decoded_representation or (decoded_representation, cross_attention_list)
        """
        x = self.embedding(tgt)
        cross_attentions = []

        for layer in self.layers:
            x, _, c_attn = layer(x, memory, src_mask, tgt_mask, return_attn=return_all_attentions)
            if return_all_attentions:
                cross_attentions.append(c_attn)

        norm_out = self.norm(x)
        if return_all_attentions:
            return norm_out, cross_attentions
        return norm_out
