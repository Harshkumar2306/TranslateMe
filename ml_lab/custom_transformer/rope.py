"""
Rotary Position Embedding (RoPE)
Su et al., 2021 (https://arxiv.org/abs/2104.09864)
Applies relative position encoding via 2D rotation matrices directly on Queries and Keys.
Enables length extrapolation, distance-aware attention, and eliminates absolute position bias.
Used in LLaMA-1/2/3, Mistral, Gemma, Falcon, PaLM.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cosine and sine frequency matrices for Rotary Position Embeddings.
    """
    assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to tensor x.
    x: (batch, n_heads, seq_len, d_k)
    cos, sin: (max_seq_len, d_k // 2)
    """
    seq_len = x.size(-2)
    d_k = x.size(-1)

    # Slice frequencies for the sequence length
    c = cos[offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device)  # (seq_len, d_k // 2)
    s = sin[offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device)  # (seq_len, d_k // 2)

    # Reshape for broadcasting with (batch, n_heads, seq_len, d_k // 2)
    c = c.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k // 2)
    s = s.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k // 2)

    # Split x into pairs of dimensions
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices

    # Apply 2D rotation:
    # [x1_rot] = [cos, -sin] [x1]
    # [x2_rot]   [sin,  cos] [x2]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c

    # Interleave back
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


class RotaryEmbedding(nn.Module):
    """
    Module holding precomputed RoPE frequency buffers.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = precompute_freqs_cis(dim, max_seq_len=max_seq_len, theta=theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to query and key tensors of shape (batch, n_heads, seq_len, d_k).
        """
        q_rot = apply_rotary_emb(q, self.cos, self.sin, offset=offset)
        k_rot = apply_rotary_emb(k, self.cos, self.sin, offset=offset)
        return q_rot, k_rot
