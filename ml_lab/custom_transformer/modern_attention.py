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

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attn: bool = False,
        offset: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Core attention computation supporting optional KV-cache for autoregressive decoding.
        q: (batch, seq_len_q, d_model)
        k: (batch, seq_len_k, d_model)
        v: (batch, seq_len_v, d_model)
        mask: Optional attention mask (boolean or float)
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)

        # 1. Project Q and apply QK-Norm
        q_proj = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        if self.use_qk_norm:
            q_proj = self.q_norm(q_proj)

        # 2. Project K, V (or reuse static cross-attention KV cache)
        if is_cross_attn and past_kv is not None:
            k_proj, v_proj = past_kv
            if self.use_rope and self.rope is not None:
                q_proj, _ = self.rope(q_proj, q_proj, offset=offset)
        else:
            seq_len_k = k.size(1)
            k_proj = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
            v_proj = self.w_v(v).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)

            if self.use_qk_norm:
                k_proj = self.k_norm(k_proj)

            # Apply RoPE at exact sequence position offset before concatenating past cache
            if self.use_rope and self.rope is not None:
                q_proj, k_proj = self.rope(q_proj, k_proj, offset=offset)

            if past_kv is not None and not is_cross_attn:
                past_k, past_v = past_kv
                k_proj = torch.cat([past_k, k_proj], dim=-2)
                v_proj = torch.cat([past_v, v_proj], dim=-2)

        present_kv = (k_proj, v_proj)
        seq_len_k_total = k_proj.size(-2)

        # 3. Attention Execution
        if return_attn:
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(q_proj.float(), k_proj.float().transpose(-2, -1)) * scale
            if mask is not None:
                m = mask.unsqueeze(1) if mask.dim() == 3 else mask
                if m.dtype == torch.bool:
                    scores = scores.masked_fill(~m, -1e9)
                else:
                    scores = scores + m.float()
            if is_causal and (seq_len_q > 1 or past_kv is None):
                causal_mask = torch.ones(
                    (seq_len_q, seq_len_k_total), dtype=torch.bool, device=q.device
                ).tril(diagonal=seq_len_k_total - seq_len_q)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            attn_weights = F.softmax(scores, dim=-1).type_as(q_proj)
            attn_weights_dropped = self.dropout(attn_weights)
            output = torch.matmul(attn_weights_dropped, v_proj)
        else:
            attn_weights = None
            attn_dropout = self.dropout_p if self.training else 0.0

            sdpa_mask = mask
            if sdpa_mask is not None and sdpa_mask.dim() == 3:
                sdpa_mask = sdpa_mask.unsqueeze(1)

            if is_causal:
                if past_kv is not None and seq_len_q == 1 and sdpa_mask is None:
                    # Single cached decode step attends to all valid cached keys without causal mask
                    output = F.scaled_dot_product_attention(
                        q_proj, k_proj, v_proj,
                        attn_mask=None,
                        dropout_p=attn_dropout,
                        is_causal=False
                    )
                elif sdpa_mask is None and past_kv is None and seq_len_q == seq_len_k_total:
                    # Pure FlashAttention / Memory-Efficient SDPA fast path
                    output = F.scaled_dot_product_attention(
                        q_proj, k_proj, v_proj,
                        attn_mask=None,
                        dropout_p=attn_dropout,
                        is_causal=True
                    )
                else:
                    # Combine explicit mask + causal mask safely (PyTorch forbids attn_mask + is_causal=True)
                    causal_bool = torch.ones(
                        (seq_len_q, seq_len_k_total), dtype=torch.bool, device=q.device
                    ).tril(diagonal=seq_len_k_total - seq_len_q).unsqueeze(0).unsqueeze(0)
                    if sdpa_mask is not None:
                        if sdpa_mask.dtype == torch.bool:
                            combined_mask = sdpa_mask & causal_bool
                        else:
                            causal_float = torch.zeros_like(causal_bool, dtype=sdpa_mask.dtype)
                            causal_float.masked_fill_(~causal_bool, -1e4)
                            combined_mask = sdpa_mask + causal_float
                    else:
                        combined_mask = causal_bool
                    output = F.scaled_dot_product_attention(
                        q_proj, k_proj, v_proj,
                        attn_mask=combined_mask,
                        dropout_p=attn_dropout,
                        is_causal=False
                    )
            else:
                output = F.scaled_dot_product_attention(
                    q_proj, k_proj, v_proj,
                    attn_mask=sdpa_mask,
                    dropout_p=attn_dropout,
                    is_causal=False
                )

        # 4. Concatenate heads and final projection: (batch, seq_len_q, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.w_o(output)

        if return_attn:
            return output, attn_weights, present_kv
        return output, None, present_kv

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
        Standard forward pass returning (output, optional_attn_weights).
        """
        output, attn_weights, _ = self._compute_attention(
            q, k, v,
            mask=mask,
            is_causal=is_causal,
            return_attn=return_attn,
            offset=offset
        )
        return output, attn_weights

    def forward_cached(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_attn: bool = False,
        offset: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        KV-cached forward pass for fast O(L) autoregressive decoding.
        Returns (output, optional_attn_weights, present_kv).
        """
        return self._compute_attention(
            q, k, v,
            mask=mask,
            is_causal=is_causal,
            return_attn=return_attn,
            offset=offset,
            past_kv=past_kv,
            is_cross_attn=is_cross_attn
        )
