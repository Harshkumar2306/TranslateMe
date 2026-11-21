"""
Complete Sequence-to-Sequence Transformer Architecture
Section 3 of 'Attention Is All You Need' (Vaswani et al., 2017)
Independent, from-scratch PyTorch implementation with weight tying,
beam search decoding, target control conditioning, and attention extraction.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .embeddings import TransformerEmbedding


@dataclass
class TransformerConfig:
    """Configurable hyperparameters for Transformer architecture."""
    vocab_size: int = 32000
    src_vocab_size: Optional[int] = None
    tgt_vocab_size: Optional[int] = None
    d_model: int = 512
    n_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    max_len: int = 512
    dropout: float = 0.1
    pad_idx: int = 0
    share_embeddings: bool = True
    pre_ln: bool = False

    def __post_init__(self):
        if self.src_vocab_size is None:
            self.src_vocab_size = self.vocab_size
        if self.tgt_vocab_size is None:
            self.tgt_vocab_size = self.vocab_size

    @classmethod
    def base(cls, vocab_size: int = 32000) -> "TransformerConfig":
        """Standard Vaswani et al. Base configuration."""
        return cls(
            vocab_size=vocab_size,
            d_model=512,
            n_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1
        )

    @classmethod
    def small(cls, vocab_size: int = 16000) -> "TransformerConfig":
        """Compact configuration suitable for fast experimentation & edge inference."""
        return cls(
            vocab_size=vocab_size,
            d_model=256,
            n_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            d_ff=1024,
            dropout=0.1
        )

    @classmethod
    def tiny(cls, vocab_size: int = 1000) -> "TransformerConfig":
        """Ultra-lightweight configuration for smoke tests, unit tests, and debugging."""
        return cls(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128,
            dropout=0.0
        )


class Generator(nn.Module):
    """
    Standard linear projection followed by log-softmax layer
    to convert decoder output vectors to token probability distributions.
    Section 3.4 of 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, vocab_size: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=bias)

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
    from attending to subsequent tokens (look-ahead masking).
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
    Full Encoder-Decoder Multilingual Transformer Architecture.
    
    Mathematical Formulation:
      Memory = Encoder(Source, Src_Mask)
      Output = Decoder(Target, Memory, Src_Mask, Tgt_Mask)
      P(w)   = Softmax(Linear(Output))
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        share_embeddings: bool = True,
        pre_ln: bool = False
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.share_embeddings = share_embeddings

        # Shared multilingual embedding or separate tables
        if share_embeddings and src_vocab_size == tgt_vocab_size:
            shared_lut = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
            enc_embedding = TransformerEmbedding(
                src_vocab_size, d_model, max_len=max_len, dropout=dropout,
                padding_idx=pad_idx, shared_embedding=shared_lut
            )
            dec_embedding = TransformerEmbedding(
                tgt_vocab_size, d_model, max_len=max_len, dropout=dropout,
                padding_idx=pad_idx, shared_embedding=shared_lut
            )
        else:
            shared_lut = None
            enc_embedding = TransformerEmbedding(
                src_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx
            )
            dec_embedding = TransformerEmbedding(
                tgt_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx
            )

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_layers=num_encoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pre_ln=pre_ln,
            embedding=enc_embedding
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=num_decoder_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pre_ln=pre_ln,
            embedding=dec_embedding
        )

        self.generator = Generator(d_model, tgt_vocab_size)

        # Weight tying: share projection weights with embedding weights (Section 3.4)
        if share_embeddings and shared_lut is not None:
            self.generator.proj.weight = shared_lut.weight

        # Parameter initialization: Xavier Uniform
        self._init_weights()

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "Seq2SeqTransformer":
        return cls(
            src_vocab_size=config.src_vocab_size or config.vocab_size,
            tgt_vocab_size=config.tgt_vocab_size or config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            d_ff=config.d_ff,
            max_len=config.max_len,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
            share_embeddings=config.share_embeddings,
            pre_ln=config.pre_ln
        )

    def _init_weights(self):
        """Initializes model parameters using Xavier uniform distribution."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "proj.weight" not in name:
                nn.init.xavier_uniform_(p)

    def count_parameters(self) -> Dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> torch.Tensor:
        return self.encoder(src, src_mask, return_all_attentions=return_all_attentions)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_all_attentions: bool = False
    ) -> torch.Tensor:
        return self.decoder(tgt, memory, src_mask, tgt_mask, return_all_attentions=return_all_attentions)

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
        max_len: int = 64,
        prefix_tokens: Optional[List[int]] = None,
        return_attention: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        disallowed_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressively generates target tokens using Greedy Search.
        Supports target language prefix conditioning, anti-hallucination n-gram blocking,
        and script-level vocabulary constraints.
        """
        self.eval()
        device = src.device
        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask)

        # Initialize target sequence
        if prefix_tokens is not None and len(prefix_tokens) > 0:
            initial_seq = [sos_idx] + prefix_tokens
            ys = torch.tensor([initial_seq], dtype=torch.long, device=device)
        else:
            ys = torch.ones(src.size(0), 1, dtype=torch.long, device=device).fill_(sos_idx)

        last_cross_attn = None

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, self.pad_idx)
            if return_attention:
                out, cross_attns = self.decode(ys, memory, src_mask, tgt_mask, return_all_attentions=True)
                if cross_attns:
                    last_cross_attn = cross_attns[-1]  # Last decoder layer cross attention
            else:
                out = self.decode(ys, memory, src_mask, tgt_mask)

            log_prob = self.generator(out[:, -1:]).view(-1).clone()  # (vocab_size,)

            # 1. Apply script-level mask if provided
            if disallowed_mask is not None:
                log_prob.masked_fill_(disallowed_mask, -float("inf"))

            # 2. Block immediate consecutive repeat
            if ys.size(1) > 1:
                last_tok = ys[0, -1].item()
                log_prob[last_tok] = -float("inf")

            # 3. Block repeated n-grams
            seq_list = ys[0].tolist()
            if no_repeat_ngram_size > 0 and len(seq_list) >= no_repeat_ngram_size:
                prefix = tuple(seq_list[-(no_repeat_ngram_size - 1):])
                for i in range(len(seq_list) - no_repeat_ngram_size + 1):
                    if tuple(seq_list[i:i + no_repeat_ngram_size - 1]) == prefix:
                        log_prob[seq_list[i + no_repeat_ngram_size - 1]] = -float("inf")

            # 4. Standard repetition penalty for all previously generated tokens
            if repetition_penalty > 1.0 and ys.size(1) > 1:
                for token_id in set(seq_list):
                    if token_id == sos_idx or token_id == self.pad_idx:
                        continue
                    if log_prob[token_id] != -float("inf"):
                        if log_prob[token_id] < 0:
                            log_prob[token_id] *= repetition_penalty
                        else:
                            log_prob[token_id] /= repetition_penalty

            next_token = torch.argmax(log_prob).view(1, 1)
            ys = torch.cat([ys, next_token], dim=1)

            if next_token.item() == eos_idx:
                break

        if return_attention:
            return ys, last_cross_attn
        return ys

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 64,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.25,
        no_repeat_ngram_size: int = 2,
        sos_idx: int = 1,
        eos_idx: int = 2,
        prefix_tokens: Optional[List[int]] = None,
        disallowed_mask: Optional[torch.Tensor] = None
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Beam search decoding with strict anti-hallucination safeguards:
        - Script masking: disallows foreign alphabet subwords
        - N-gram blocking: prevents cyclic loops (e.g. repeated commas or phrases)
        - Immediate repeat suppression
        """
        self.eval()
        device = src.device
        src_mask = make_src_mask(src, self.pad_idx)
        memory = self.encode(src, src_mask)

        initial_seq = [sos_idx]
        if prefix_tokens:
            initial_seq += prefix_tokens

        # Hypothesis: (sequence_tensor, cumulative_log_prob)
        beams = [(torch.tensor([initial_seq], dtype=torch.long, device=device), 0.0)]
        completed_beams = []

        for step in range(max_len):
            new_candidates = []

            for seq, score in beams:
                tgt_mask = make_tgt_mask(seq, self.pad_idx)
                out = self.decode(seq, memory, src_mask, tgt_mask)
                raw_log_probs = self.generator(out[:, -1:]).view(-1)  # Shape: (vocab_size,)
                log_probs = raw_log_probs.clone()

                # 1. Apply script-level mask if provided
                if disallowed_mask is not None:
                    log_probs.masked_fill_(disallowed_mask, -float("inf"))

                # 2. Block immediate consecutive repeat
                if seq.size(1) > 1:
                    last_tok = seq[0, -1].item()
                    log_probs[last_tok] = -float("inf")

                # 3. Block repeated n-grams
                seq_list = seq[0].tolist()
                if no_repeat_ngram_size > 0 and len(seq_list) >= no_repeat_ngram_size:
                    prefix = tuple(seq_list[-(no_repeat_ngram_size - 1):])
                    for i in range(len(seq_list) - no_repeat_ngram_size + 1):
                        if tuple(seq_list[i:i + no_repeat_ngram_size - 1]) == prefix:
                            log_probs[seq_list[i + no_repeat_ngram_size - 1]] = -float("inf")

                # 4. Standard repetition penalty
                if repetition_penalty > 1.0 and seq.size(1) > 1:
                    for token_id in set(seq_list):
                        if token_id == sos_idx or token_id == self.pad_idx:
                            continue
                        if log_probs[token_id] != -float("inf"):
                            if log_probs[token_id] < 0:
                                log_probs[token_id] *= repetition_penalty
                            else:
                                log_probs[token_id] /= repetition_penalty

                topk_probs, topk_indices = torch.topk(log_probs, beam_size * 2)

                for prob, idx in zip(topk_probs, topk_indices):
                    if prob.item() == -float("inf"):
                        continue
                    cand_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                    # Use unpenalized raw log prob for consistent candidate scoring
                    cand_score = score + raw_log_probs[idx].item()
                    if idx.item() == eos_idx:
                        completed_beams.append((cand_seq, cand_score))
                    else:
                        new_candidates.append((cand_seq, cand_score))

            if len(completed_beams) >= beam_size:
                break

            if not new_candidates:
                break

            # Sort candidate hypotheses by length-normalized score
            def lp_score(cand):
                s_len = cand[0].size(1)
                lp = ((5.0 + s_len) / 6.0) ** length_penalty
                return cand[1] / lp

            new_candidates.sort(key=lp_score, reverse=True)
            beams = new_candidates[:beam_size]

        # Combine completed and remaining active beams
        all_candidates = completed_beams if completed_beams else beams

        def final_lp_score(item):
            s_len = item[0].size(1)
            lp = ((5.0 + s_len) / 6.0) ** length_penalty
            return item[1] / lp

        all_candidates.sort(key=final_lp_score, reverse=True)
        return all_candidates[:beam_size]
