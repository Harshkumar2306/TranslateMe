"""
Input Embeddings and Positional Encoding
Follows 'Attention Is All You Need' (Vaswani et al., 2017)
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Standard PyTorch Embedding layer scaled by sqrt(d_model)
    as specified in Section 3.4 of 'Attention Is All You Need'.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale embedding values by sqrt(d_model) to prevent vanishing gradients
        # when adding positional encodings.
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    Since Transformer contains no recurrence and no convolution,
    positional encodings inject information about relative/absolute position:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) for broadcast addition across batches
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with position added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Combines TokenEmbedding and PositionalEncoding.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_emb = self.token_embedding(x)
        return self.positional_encoding(tok_emb)
