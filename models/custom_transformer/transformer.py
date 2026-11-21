"""
Complete Sequence-to-Sequence Transformer Architecture
Section 3 of 'Attention Is All You Need' (Vaswani et al., 2017)
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Generator(nn.Module):
    """
    Standard linear projection followed by log-softmax layer
    to convert decoder output vectors to token probability distributions.
    Section 3.4 of 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder hidden states (batch_size, seq_len, d_model)
        Returns:
            Log probabilities (batch_size, seq_len, vocab_size)
        """
        return F.log_softmax(self.proj(x), dim=-1)


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Creates a mask for padding tokens in the source sentence.
    Shape: (batch_size, 1, 1, src_len)
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size: int) -> torch.Tensor:
    """
    Creates a lower-triangular causal mask to prevent decoder positions
    from attending to subsequent tokens (look-ahead mask).
    Shape: (1, 1, size, size)
    """
    attn_shape = (1, 1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Combines target padding mask with subsequent look-ahead mask.
    Shape: (batch_size, 1, tgt_len, tgt_len)
    """
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    tgt_len = tgt.size(1)
    subsequent_mask = make_subsequent_mask(tgt_len).to(tgt.device)  # (1, 1, tgt_len, tgt_len)
    return tgt_pad_mask & subsequent_mask


class Seq2SeqTransformer(nn.Module):
    """
    Standard Encoder-Decoder Transformer Architecture.
    
    Components:
    1. Input Embeddings (Source & Target)
    2. Sinusoidal Positional Encoding
    3. Multi-Head Scaled Dot-Product Attention
    4. Position-Wise Feed-Forward Networks
    5. Residual Connections & Layer Normalization
    6. Masked Decoder Self-Attention
    7. Encoder-Decoder Cross Attention
    8. Linear Output Projection & Softmax Generation
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        d_ff: int = 512,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=num_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=num_decoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.generator = Generator(d_model, tgt_vocab_size)

        # Initialize parameters with Xavier uniform
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for teacher-forcing training.
        Args:
            src: Source token IDs (batch_size, src_len)
            tgt: Target token IDs (batch_size, tgt_len)
        Returns:
            Log probabilities (batch_size, tgt_len, tgt_vocab_size)
        """
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)

        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(dec_out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_len: int = 64
    ) -> torch.Tensor:
        """
        Autoregressively generates target tokens using Greedy Search.
        """
        self.eval()
        device = src.device
        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask)

        # Initialize target sequence with start-of-sequence token
        ys = torch.ones(src.size(0), 1, dtype=torch.long, device=device).fill_(sos_idx)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.pad_idx)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            log_prob = self.generator(out[:, -1:])
            next_token = torch.argmax(log_prob, dim=-1)  # (batch_size, 1)

            ys = torch.cat([ys, next_token], dim=1)

            # Check if all sequences in batch have reached EOS
            if (next_token == eos_idx).all():
                break

        return ys
