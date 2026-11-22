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
    use_layerscale: bool = True
    layerscale_init: float = 0.1
    soft_cap: Optional[float] = 30.0  # Gemma 2 style logit soft-capping

    @classmethod
    def modern_100m(cls, vocab_size: int = 16384) -> "ModernTransformerConfig":
        """
        Optimal ~111.7 Million Parameter SOTA Balanced Configuration:
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
            share_embeddings=True,
            use_layerscale=True,
            soft_cap=30.0
        )

    @classmethod
    def modern_asymmetric_100m(cls, vocab_size: int = 16384) -> "ModernTransformerConfig":
        """
        Asymmetric Fast-Inference 100M Configuration (~106.9M Parameters):
        8 Deep Encoder Layers for rich source semantics, 4 Fast Decoder Layers for 33% lower latency.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=768,
            n_heads=12,
            num_encoder_layers=8,
            num_decoder_layers=4,
            d_ff=2048,
            dropout=0.1,
            use_qk_norm=True,
            use_rope=True,
            share_embeddings=True,
            use_layerscale=True,
            soft_cap=30.0
        )

    @classmethod
    def modern_125m(cls, vocab_size: int = 16384) -> "ModernTransformerConfig":
        """
        Deep-Capacity Configuration (~125.8M Parameters):
        8 Encoder Layers, 6 Decoder Layers.
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
            share_embeddings=True,
            use_layerscale=True,
            soft_cap=30.0
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
            share_embeddings=True,
            use_layerscale=True,
            soft_cap=30.0
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

        # 2. Modern Encoder Stack with LayerScale
        self.encoder_layers = nn.ModuleList([
            ModernEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_qk_norm=config.use_qk_norm,
                use_rope=config.use_rope,
                use_layerscale=config.use_layerscale,
                layerscale_init=config.layerscale_init
            )
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_final_norm = RMSNorm(config.d_model)

        # 3. Modern Decoder Stack with LayerScale
        self.decoder_layers = nn.ModuleList([
            ModernDecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_qk_norm=config.use_qk_norm,
                use_rope=config.use_rope,
                use_layerscale=config.use_layerscale,
                layerscale_init=config.layerscale_init
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

        # Gemma 2 style logit soft-capping to prevent overconfident runaway loops
        if self.config.soft_cap is not None and self.config.soft_cap > 0:
            logits = self.config.soft_cap * torch.tanh(logits / self.config.soft_cap)

        return logits.contiguous(), cross_attn_list

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
        if src.numel() == 0 or tgt.numel() == 0:
            raise ValueError("Input tensors 'src' and 'tgt' cannot be empty.")

        device = self.shared_embedding.weight.device
        if src.device != device:
            src = src.to(device)
        if tgt.device != device:
            tgt = tgt.to(device)

        if src_mask is None:
            src_mask = self.make_src_mask(src)
        elif src_mask.device != device:
            src_mask = src_mask.to(device)

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
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 0.6,
        script_mask: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[List[int]] = None
    ) -> List[List[int]]:
        """
        Autoregressive sequence generation with Greedy or Beam Search.
        Includes Gemma 2 logit soft-capping, script masking, 3-gram repetition blocking,
        and robust finished-sequence clamping for batch inference.
        src: (batch, src_len)
        """
        self.eval()
        batch_size = src.size(0)
        if src.numel() == 0 or src.size(1) == 0:
            return [[] for _ in range(batch_size)]
        if max_len <= 0:
            return [[self.bos_idx] for _ in range(batch_size)]

        device = self.shared_embedding.weight.device
        if src.device != device:
            src = src.to(device)

        num_beams = max(1, int(num_beams))

        src_mask = self.make_src_mask(src)
        memory, _ = self.encode(src, src_mask=src_mask, return_attn=False)

        if prefix_tokens is not None and len(prefix_tokens) > 0:
            init_tokens = [self.bos_idx] + prefix_tokens
        else:
            init_tokens = [self.bos_idx]

        if num_beams == 1:
            # Greedy decoding with batch-correct finished sequence clamping
            generated = torch.tensor([init_tokens] * batch_size, dtype=torch.long, device=device)
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

                # Correctness fix: finished sequences are clamped to pad_idx
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, self.pad_idx),
                    next_tokens
                )
                finished = finished | (next_tokens.squeeze(-1) == self.eos_idx)
                generated = torch.cat([generated, next_tokens], dim=-1)

                if finished.all():
                    break

            # Strip trailing padding tokens per sentence
            results = []
            for seq in generated.tolist():
                clean_seq = []
                for tok in seq:
                    clean_seq.append(tok)
                    if tok == self.eos_idx:
                        break
                results.append(clean_seq)
            return results
        else:
            # Beam Search with completed hypothesis separation and length normalization
            results = []
            for b in range(batch_size):
                b_mem = memory[b:b+1]  # (1, src_len, d_model)
                b_mask = src_mask[b:b+1]

                beams = [(init_tokens, 0.0)]
                completed: List[Tuple[List[int], float]] = []

                for _ in range(max_len):
                    all_candidates = []

                    for seq, score in beams:
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

                        # N-gram repetition blocking (prevents repeating loops like ", , ,")
                        if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size - 1:
                            ngram_prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                            for i in range(len(seq) - no_repeat_ngram_size + 1):
                                if tuple(seq[i : i + no_repeat_ngram_size - 1]) == ngram_prefix:
                                    banned_tok = seq[i + no_repeat_ngram_size - 1]
                                    next_logits[banned_tok] = -1e4

                        if script_mask is not None:
                            next_logits.masked_fill_(script_mask, -1e4)

                        log_probs = F.log_softmax(next_logits, dim=-1)
                        topk_probs, topk_tokens = torch.topk(log_probs, num_beams * 2)

                        for lp, tok in zip(topk_probs.tolist(), topk_tokens.tolist()):
                            new_seq = seq + [tok]
                            new_score = score + lp
                            if tok == self.eos_idx:
                                completed.append((new_seq, new_score))
                            else:
                                all_candidates.append((new_seq, new_score))

                    def score_fn(candidate):
                        seq, score = candidate
                        # Length penalty formula: ((5 + len) / 6) ^ alpha
                        lp = ((5.0 + len(seq)) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                        return score / lp

                    # Robust early stopping: break only if completed candidates dominate active beams
                    if len(completed) >= num_beams:
                        best_active = max(score_fn(c) for c in all_candidates) if all_candidates else -float("inf")
                        worst_completed = min(score_fn(c) for c in completed[:num_beams])
                        if worst_completed >= best_active:
                            break

                    ordered = sorted(all_candidates, key=score_fn, reverse=True)
                    beams = ordered[:num_beams]
                    if not beams:
                        break

                # Combine completed hypotheses with any surviving beams
                pool = completed if completed else beams
                def final_score_fn(candidate):
                    seq, score = candidate
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
                    return score / lp

                ordered_final = sorted(pool, key=final_score_fn, reverse=True)
                best_seq = ordered_final[0][0]
                results.append(best_seq)

            return results

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_idx: int = 2,
        eos_idx: int = 3,
        max_len: int = 64,
        prefix_tokens: Optional[List[int]] = None,
        return_attention: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        disallowed_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Polymorphic greedy decoding wrapper matching classical Seq2SeqTransformer signature."""
        prefix = [sos_idx] + (prefix_tokens or [])
        res = self.generate(
            src=src,
            max_len=max_len,
            num_beams=1,
            prefix_tokens=prefix,
            return_attention=return_attention,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            disallowed_mask=disallowed_mask
        )
        if return_attention:
            tokens, attn = res
            return tokens, attn
        return res, None

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 64,
        sos_idx: int = 2,
        eos_idx: int = 3,
        prefix_tokens: Optional[List[int]] = None,
        disallowed_mask: Optional[torch.Tensor] = None,
        no_repeat_ngram_size: int = 2,
        repetition_penalty: float = 1.0
    ) -> List[Tuple[torch.Tensor, float]]:
        """Polymorphic beam search decoding wrapper returning [(seq_tensor, score)] matching classical signature."""
        prefix = [sos_idx] + (prefix_tokens or [])
        res = self.generate(
            src=src,
            max_len=max_len,
            num_beams=beam_size,
            prefix_tokens=prefix,
            return_attention=False,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            disallowed_mask=disallowed_mask
        )
        out = []
        for seq in res:
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            out.append((seq, 0.0))
        return out
