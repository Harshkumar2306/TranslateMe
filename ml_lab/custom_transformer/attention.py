"""
Scaled Dot-Product Attention and Multi-Head Attention Modules
Section 3.2 of 'Attention Is All You Need' (Vaswani et al., 2017)
Implemented strictly from first principles using PyTorch tensors.
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
    
    Properties:
    - Dot products grow large in magnitude for large d_k, pushing softmax into small gradients.
    - We scale by 1 / sqrt(d_k) to counteract this effect.
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
            query: Tensor of shape (batch_size, n_heads, seq_len_q, d_k)
            key:   Tensor of shape (batch_size, n_heads, seq_len_k, d_k)
            value: Tensor of shape (batch_size, n_heads, seq_len_v, d_v)
            mask:  Optional broadcastable mask (batch_size, 1, seq_len_q, seq_len_k)
                   Positions where mask == 0 or False will be masked out with -1e9.
        Returns:
            output: Attention output (batch_size, n_heads, seq_len_q, d_v)
            attn_weights: Attention matrix (batch_size, n_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)

        # 1. Compute raw affinity scores: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply attention mask if provided
        if mask is not None:
            # Mask positions with a very large negative number so softmax evaluates to zero
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax across key sequence dimension (sum to 1 per query position)
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
    
    Linearly projects queries, keys, and values h times with different,
    learned linear projections to d_k, d_k, and d_v dimensions respectively.
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

        # Final output projection W^O
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            q: Queries tensor (batch_size, seq_len_q, d_model)
            k: Keys tensor    (batch_size, seq_len_k, d_model)
            v: Values tensor  (batch_size, seq_len_v, d_model)
            mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
            return_attn: Whether to return the attention probability matrix
        Returns:
            output or (output, attn_weights)
        """
        batch_size = q.size(0)

        # 1. Project through W_q, W_k, W_v and reshape into heads:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        attn_out, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=mask)
        self.last_attn_weights = attn_weights.detach()

        # 3. Concatenate heads and project through W^O
        # (batch_size, n_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, n_heads * d_k)
        concat = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(concat)

        if return_attn:
            return output, attn_weights
        return output
