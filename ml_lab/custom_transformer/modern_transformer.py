"""
Modern SOTA Seq2Seq Transformer Architecture (100M+ Scale)
Translates across English, Hindi, Tamil, and Telugu without pretrained weights.

Key Architectural Innovations:
- Pre-LN with RMSNorm (No mean-centering overhead, infinite gradient highway)
- Rotary Position Embeddings (RoPE) for relative token-distance awareness
- SwiGLU Gated Feed-Forward Networks for maximum parameter expressiveness
- QK-Norm (RMSNorm on Queries and Keys) to eliminate attention logit explosion
- PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention-2) kernel fusion
- Three-Way Weight Tying (Encoder Embedding = Decoder Embedding = Projection Matrix)
- Dynamic Beam Search with Length Penalty, Repetition Penalty, and Script Masking
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .modern_layers import ModernEncoderLayer, ModernDecoderLayer


@dataclass
class ModernTransformerConfig:
    """Hyperparameters for Modern SOTA Transformer."""
    vocab_size: int = 16384
    d_model: int = 768
    n_heads: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    max_len: int = 1024
    dropout: float = 0.1
    pad_idx: int = 0
    bos_idx: int = 2
    eos_idx: int = 3
    use_qk_norm: bool = True
    use_rope: bool = True
    share_embeddings: bool = True

    @classmethod
    def modern_100m(cls, vocab_size: int = 16384) -> "ModernTransformerConfig":
        """
        Optimal ~111.7 Million Parameter SOTA Configuration:
        6 Encoder Layers, 6 Decoder Layers, d_model=768, 12 Heads, SwiGLU d_ff=2048.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_heads=12,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            use_qk_norm=True,
            use_rope=True,
            share_embeddings=True
        )

    @classmethod
    def modern_125m(cls, vocab_size: int = 16384) -> "ModernTransformerConfig":
        """
        Asymmetric Deep-Encoder Configuration (~125.8M Parameters):
        8 Encoder Layers, 6 Decoder Layers for deeper source representation.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_heads=12,
            num_encoder_layers=8,
            num_decoder_layers=6,
            d_ff=2048,
            dropout=0.1,
            use_qk_norm=True,
            use_rope=True,
            share_embeddings=True
        )

    @classmethod
    def modern_small(cls, vocab_size: int = 8192) -> "ModernTransformerConfig":
        """Compact 25M configuration for fast local smoke testing."""
        return cls(
            vocab_size=vocab_size,
            d_model=384,
            n_heads=6,
            num_encoder_layers=4,
            num_decoder_layers=4,
            d_ff=1024,
            dropout=0.1,
            use_qk_norm=True,
            use_rope=True,
            share_embeddings=True
        )


class ModernSeq2SeqTransformer(nn.Module):
    """
    Modern 100M+ SOTA Multilingual Transformer (Seq2Seq).
    """
    def __init__(self, config: ModernTransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.pad_idx = config.pad_idx
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx

        # 1. Token Embeddings (Scaled by sqrt(d_model))
        self.shared_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_idx
        )
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        # 2. Modern Encoder Stack
        self.encoder_layers = nn.ModuleList([
            ModernEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_qk_norm=config.use_qk_norm,
                use_rope=config.use_rope
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_norm = RMSNorm(config.d_model)

        # 3. Modern Decoder Stack
        self.decoder_layers = nn.ModuleList([
            ModernDecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_qk_norm=config.use_qk_norm,
                use_rope=config.use_rope
            )
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_final_norm = RMSNorm(config.d_model)

        # 4. Output Projection Layer (Tied with Shared Embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.share_embeddings:
            self.lm_head.weight = self.shared_embedding.weight

        # Initialize weights with Xavier uniform scaled for depth
        self._init_weights()

    def _init_weights(self):
        """Initializes linear and embedding weights for stable deep training."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name and not isinstance(p, nn.Embedding):
                nn.init.xavier_uniform_(p)
            elif isinstance(p, nn.Embedding):
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(self.d_model))

    def count_parameters(self) -> int:
        """Returns total number of unique trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Creates padding mask for encoder: (batch, 1, seq_len)
        True for real tokens, False for padding.
        """
        return (src != self.pad_idx).unsqueeze(1)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Encodes source token IDs into continuous hidden representations.
        src: (batch, src_len)
        Returns: memory (batch, src_len, d_model)
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        x = self.shared_embedding(src) * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        attn_list = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask=src_mask, return_attn=return_attn)
            if return_attn:
                attn_list.append(attn)

        memory = self.encoder_final_norm(x)
        return memory, attn_list

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Decodes target token IDs given encoder memory.
        tgt: (batch, tgt_len)
        memory: (batch, src_len, d_model)
        """
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        cross_attn_list = []
        for layer in self.decoder_layers:
            x, cross_attn = layer(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                is_causal=is_causal,
                return_cross_attn=return_cross_attn
            )
            if return_cross_attn:
                cross_attn_list.append(cross_attn)

        x = self.decoder_final_norm(x)
        logits = self.lm_head(x)
        return logits, cross_attn_list

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full teacher-forcing forward pass for training.
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        Returns: logits (batch, tgt_len, vocab_size)
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        memory, _ = self.encode(src, src_mask=src_mask, return_attn=False)
        logits, _ = self.decode_step(
            tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            is_causal=True,
            return_cross_attn=False
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 128,
        num_beams: int = 1,
        repetition_penalty: float = 1.15,
        length_penalty: float = 0.6,
        script_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Autoregressive sequence generation with Greedy or Beam Search.
        src: (batch, src_len)
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        src_mask = self.make_src_mask(src)
        memory, _ = self.encode(src, src_mask=src_mask, return_attn=False)

        if num_beams == 1:
            # Greedy decoding
            generated = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                logits, _ = self.decode_step(
                    generated,
                    memory=memory,
                    memory_mask=src_mask,
                    is_causal=True
                )
                next_token_logits = logits[:, -1, :].clone()

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for b in range(batch_size):
                        for tok in set(generated[b].tolist()):
                            if tok != self.pad_idx and tok != self.bos_idx:
                                if next_token_logits[b, tok] > 0:
                                    next_token_logits[b, tok] /= repetition_penalty
                                else:
                                    next_token_logits[b, tok] *= repetition_penalty

                # Mask foreign scripts if provided
                if script_mask is not None:
                    next_token_logits.masked_fill_(script_mask, -1e4)

                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tokens], dim=-1)

                finished = finished | (next_tokens.squeeze(-1) == self.eos_idx)
                if finished.all():
                    break

            return generated.tolist()

        else:
            # Beam Search
            # For each item in batch, run beam search
            results = []
            for b in range(batch_size):
                b_mem = memory[b:b+1]  # (1, src_len, d_model)
                b_mask = src_mask[b:b+1]

                # Initialize beam: (seq, log_prob, finished)
                beams = [([self.bos_idx], 0.0, False)]

                for _ in range(max_len):
                    all_candidates = []
                    all_finished = True

                    for seq, score, done in beams:
                        if done:
                            all_candidates.append((seq, score, True))
                            continue

                        all_finished = False
                        input_tensor = torch.tensor([seq], dtype=torch.long, device=device)
                        logits, _ = self.decode_step(
                            input_tensor,
                            memory=b_mem,
                            memory_mask=b_mask,
                            is_causal=True
                        )
                        next_logits = logits[0, -1, :].clone()

                        # Repetition penalty
                        if repetition_penalty != 1.0:
                            for tok in set(seq):
                                if tok not in (self.pad_idx, self.bos_idx):
                                    if next_logits[tok] > 0:
                                        next_logits[tok] /= repetition_penalty
                                    else:
                                        next_logits[tok] *= repetition_penalty

                        if script_mask is not None:
                            next_logits.masked_fill_(script_mask, -1e4)

                        log_probs = F.log_softmax(next_logits, dim=-1)
                        topk_probs, topk_tokens = torch.topk(log_probs, num_beams * 2)

                        for lp, tok in zip(topk_probs.tolist(), topk_tokens.tolist()):
                            new_seq = seq + [tok]
                            new_score = score + lp
                            new_done = (tok == self.eos_idx)
                            all_candidates.append((new_seq, new_score, new_done))

                    if all_finished:
                        break

                    # Rank by length-normalized score
                    def score_fn(candidate):
                        seq, score, done = candidate
                        # Length penalty formula: ((5 + len) / 6) ^ alpha
                        lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                        return score / lp

                    ordered = sorted(all_candidates, key=score_fn, reverse=True)
                    beams = ordered[:num_beams]

                best_seq = beams[0][0]
                results.append(best_seq)

            return results
