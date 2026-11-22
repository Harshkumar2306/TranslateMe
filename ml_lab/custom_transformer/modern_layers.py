"""
Modern Encoder and Decoder Layers with Pre-LN RMSNorm, SwiGLU, and QK-Norm
Pre-LN architecture guarantees stable gradient flow in deep networks (8 to 32 layers).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .swiglu import SwiGLU
from .modern_attention import ModernMultiHeadAttention


class ModernEncoderLayer(nn.Module):
    """
    Modern Transformer Encoder Layer:
    x -> RMSNorm -> ModernMHA (Self) -> Residual -> RMSNorm -> SwiGLU FFN -> Residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            norm_x, norm_x, norm_x,
            mask=mask,
            is_causal=False,
            return_attn=return_attn
        )
        x = x + self.dropout(attn_out)

        # Pre-LN SwiGLU Feed-Forward
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x, attn_weights


class ModernDecoderLayer(nn.Module):
    """
    Modern Transformer Decoder Layer:
    x -> RMSNorm -> Causal Self-Attn -> Residual
      -> RMSNorm -> Cross-Attn (Memory) -> Residual
      -> RMSNorm -> SwiGLU FFN -> Residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope
        )

        self.norm2 = RMSNorm(d_model)
        self.cross_attn = ModernMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_qk_norm=use_qk_norm,
            use_rope=False  # RoPE typically not applied to cross-attention between different sequences
        )

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Pre-LN Causal Self-Attention
        norm_x = self.norm1(x)
        self_out, _ = self.self_attn(
            norm_x, norm_x, norm_x,
            mask=tgt_mask,
            is_causal=is_causal,
            return_attn=False
        )
        x = x + self.dropout(self_out)

        # 2. Pre-LN Cross-Attention over Encoder Memory
        norm_x = self.norm2(x)
        cross_out, cross_weights = self.cross_attn(
            norm_x, memory, memory,
            mask=memory_mask,
            is_causal=False,
            return_attn=return_cross_attn
        )
        x = x + self.dropout(cross_out)

        # 3. Pre-LN SwiGLU Feed-Forward
        x = x + self.dropout(self.ffn(self.norm3(x)))

        return x, cross_weights
