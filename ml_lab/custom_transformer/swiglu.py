"""
Swish Gated Linear Unit (SwiGLU) Feed-Forward Network
Shazeer, 2020 (https://arxiv.org/abs/2002.05202)
Replaces classical ReLU MLP with a bilinear gated activation:
SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
Proven to significantly improve perplexity, representation capacity,
and convergence speed per parameter.
Standard in modern LLMs: LLaMA-1/2/3, PaLM, Mistral, Gemma.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network:
    Input: (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)
    """
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        # Standard scaling: d_ff is approx 8/3 * d_model or 2/3 * 4 * d_model
        if d_ff is None:
            # Round to multiple of 64 for GPU tensor core efficiency
            raw_dff = int(2 * (4 * d_model) / 3)
            d_ff = ((raw_dff + 63) // 64) * 64

        self.d_model = d_model
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.silu(x * w_gate) * (x * w_up)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.w_down(hidden)
        return out

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, d_ff={self.d_ff}"
