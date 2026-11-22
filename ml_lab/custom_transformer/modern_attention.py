"""
Modern Multi-Head Attention with QK-Norm and PyTorch SDPA (FlashAttention-2)
Incorporates:
1. QK-Norm (RMSNorm on Queries and Keys) to eliminate attention logit explosion and loss spikes
2. Rotary Position Embeddings (RoPE) for relative position awareness
3. PyTorch F.scaled_dot_product_attention (SDPA) for FlashAttention-2 kernel execution
4. Backward compatibility for attention weight visualization in DL Lab
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .rope import RotaryEmbedding


class ModernMultiHeadAttention(nn.Module):
    """
    State-of-the-art Multi-Head Attention with:
    - QK-Norm (Dehghani et al., 2023)
    - Optional Rotary Position Embedding (RoPE)
    - PyTorch Fused SDPA (FlashAttention / Memory Efficient)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_qk_norm: bool = True,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        # QK-Norm applies RMSNorm to each attention head vector
        if use_qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Rotary Position Embedding
        if use_rope:
            self.rope = RotaryEmbedding(dim=self.d_k, max_seq_len=rope_max_seq_len)
        else:
            self.rope = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attn: bool = False,
        offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        q: (batch, seq_len_q, d_model)
        k: (batch, seq_len_k, d_model)
        v: (batch, seq_len_v, d_model)
        mask: Optional attention mask (boolean or float)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # 1. Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        q_proj = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

        # 2. QK-Norm (RMSNorm along head dimension d_k)
        if self.use_qk_norm:
            q_proj = self.q_norm(q_proj)
            k_proj = self.k_norm(k_proj)

        # 3. Apply Rotary Position Embedding if enabled
        if self.use_rope and self.rope is not None:
            q_proj, k_proj = self.rope(q_proj, k_proj, offset=offset)

        # 4. Attention Execution
        # If explicit attention matrix is requested (for research/visualization):
        if return_attn:
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * scale
            if mask is not None:
                # Safe FP16 mask fill
                if mask.dtype == torch.bool:
                    scores = scores.masked_fill(~mask, -1e4)
                else:
                    scores = scores + mask
            if is_causal:
                causal_mask = torch.triu(torch.full((seq_len_q, seq_len_k), float('-inf'), device=q.device), diagonal=1)
                scores = scores + causal_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights_dropped = self.dropout(attn_weights)
            output = torch.matmul(attn_weights_dropped, v_proj)
        else:
            attn_weights = None
            attn_dropout = self.dropout_p if self.training else 0.0

            # Convert 3D mask (batch, 1, seq_k) to 4D (batch, 1, 1, seq_k) if needed
            sdpa_mask = mask
            if sdpa_mask is not None and sdpa_mask.dim() == 3:
                sdpa_mask = sdpa_mask.unsqueeze(1)

            # PyTorch SDPA (dispatches to FlashAttention-2 / Cutlass / C++ fast path)
            output = F.scaled_dot_product_attention(
                q_proj,
                k_proj,
                v_proj,
                attn_mask=sdpa_mask,
                dropout_p=attn_dropout,
                is_causal=is_causal
            )

        # 5. Concatenate heads and final projection: (batch, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(output)

        if return_attn:
            return output, attn_weights
        return output, None
