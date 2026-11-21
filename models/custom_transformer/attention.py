"""
Scaled Dot-Product Attention and Multi-Head Attention
Section 3.2 of 'Attention Is All You Need'
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Shape (batch_size, n_heads, seq_len_q, d_k)
            key:   Shape (batch_size, n_heads, seq_len_k, d_k)
            value: Shape (batch_size, n_heads, seq_len_v, d_v)
            mask:  Optional mask tensor (batch_size, 1, seq_len_q, seq_len_k)
        Returns:
            output: Attention output (batch_size, n_heads, seq_len_q, d_v)
            attn_weights: Attention probabilities (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)

        # 1. Compute raw affinity scores: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply attention mask (e.g. padding mask or causal look-ahead mask)
        if mask is not None:
            # Masked positions get large negative value (-1e9) so softmax evaluates to ~0
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax across key sequence length dimension
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        # 4. Multiply with values
        output = torch.matmul(attn_weights_dropped, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
      where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    Allows model to jointly attend to information from different representation subspaces.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: Queries tensor (batch_size, seq_len_q, d_model)
            k: Keys tensor    (batch_size, seq_len_k, d_model)
            v: Values tensor  (batch_size, seq_len_v, d_model)
            mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
        Returns:
            Projected attention tensor (batch_size, seq_len_q, d_model)
        """
        batch_size = q.size(0)

        # 1. Linear projection and split into heads:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention across all heads simultaneously
        attn_out, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=mask)
        self.last_attn_weights = attn_weights.detach()

        # 3. Concatenate heads and project through W^O:
        # (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads * d_k)
        concat = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(concat)
