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
    Modern Transformer Encoder Layer with Pre-LN RMSNorm, SwiGLU, and LayerScale:
    x -> RMSNorm -> ModernMHA (Self) -> LayerScale -> Residual
      -> RMSNorm -> SwiGLU FFN -> LayerScale -> Residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_layerscale: bool = True,
        layerscale_init: float = 0.1
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

        self.use_layerscale = use_layerscale
        if use_layerscale:
            self.gamma_1 = nn.Parameter(torch.ones(d_model) * layerscale_init)
            self.gamma_2 = nn.Parameter(torch.ones(d_model) * layerscale_init)
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention with LayerScale
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            norm_x, norm_x, norm_x,
            mask=mask,
            is_causal=False,
            return_attn=return_attn
        )
        x = x + self.dropout(self.gamma_1 * attn_out)

        # Pre-LN SwiGLU Feed-Forward with LayerScale
        x = x + self.dropout(self.gamma_2 * self.ffn(self.norm2(x)))

        return x, attn_weights


class ModernDecoderLayer(nn.Module):
    """
    Modern Transformer Decoder Layer with LayerScale:
    x -> RMSNorm -> Causal Self-Attn -> LayerScale -> Residual
      -> RMSNorm -> Cross-Attn (Memory) -> LayerScale -> Residual
      -> RMSNorm -> SwiGLU FFN -> LayerScale -> Residual
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_layerscale: bool = True,
        layerscale_init: float = 0.1
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
            use_rope=False  # RoPE typically not applied to cross-attention
        )

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.use_layerscale = use_layerscale
        if use_layerscale:
            self.gamma_1 = nn.Parameter(torch.ones(d_model) * layerscale_init)
            self.gamma_2 = nn.Parameter(torch.ones(d_model) * layerscale_init)
            self.gamma_3 = nn.Parameter(torch.ones(d_model) * layerscale_init)
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0
            self.gamma_3 = 1.0

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Pre-LN Causal Self-Attention with LayerScale
        norm_x = self.norm1(x)
        self_out, _ = self.self_attn(
            norm_x, norm_x, norm_x,
            mask=tgt_mask,
            is_causal=is_causal,
            return_attn=False
        )
        x = x + self.dropout(self.gamma_1 * self_out)

        # 2. Pre-LN Cross-Attention over Encoder Memory with LayerScale
        norm_x = self.norm2(x)
        cross_out, cross_weights = self.cross_attn(
            norm_x, memory, memory,
            mask=memory_mask,
            is_causal=False,
            return_attn=return_cross_attn
        )
        x = x + self.dropout(self.gamma_2 * cross_out)

        # 3. Pre-LN SwiGLU Feed-Forward with LayerScale
        x = x + self.dropout(self.gamma_3 * self.ffn(self.norm3(x)))

        return x, cross_weights
