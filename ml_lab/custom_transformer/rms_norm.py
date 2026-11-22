"""
Root Mean Square Layer Normalization (RMSNorm)
Zhang & Sennrich, NeurIPS 2019 (https://arxiv.org/abs/1910.07467)
Replaces standard LayerNorm by dispensing with mean-centering, yielding
faster computation, superior numerical stability, and better gradient propagation.
Used across modern LLMs (LLaMA-3, Mistral, Gemma, PaLM).
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm(x) = (x / RMS(x)) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform normalization in float32 for mixed-precision AMP stability
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(output)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
