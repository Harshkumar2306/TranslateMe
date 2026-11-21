"""
Position-Wise Feed-Forward Network
Section 3.3 of 'Attention Is All You Need'
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
    Applied to each position separately and identically:
    Consists of two linear transformations with a ReLU (or GELU) activation in between.
    Typically d_ff = 4 * d_model (e.g. 2048 for d_model=512, or 1024 for d_model=256).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.activation == "gelu":
            h = F.gelu(self.w_1(x))
        else:
            h = F.relu(self.w_1(x))
        return self.w_2(self.dropout(h))
