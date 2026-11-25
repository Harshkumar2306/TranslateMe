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
    def modern_asymmetric_100m(cls, vocab_size: int = 24576) -> "ModernTransformerConfig":
        """
        Option 1 SOTA Asymmetric Any-to-Any Configuration (113.29M Parameters @ 24,576 vocab):
        8 Deep Encoder Layers for rich cross-lingual semantics across EN/HI/TA/TE,
        4 Fast Decoder Layers for ~33% faster autoregressive decoding.
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
        """
        Initializes embeddings, projections, and residual output layers for deep Pre-LN stability.
        Uses GPT-2 / LLaMA-style residual depth scaling on output projections (w_o, w_down).
        """
        total_depth = max(1, self.config.num_encoder_layers + self.config.num_decoder_layers)
        residual_scale = 1.0 / math.sqrt(2.0 * total_depth)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if name.endswith(("w_o", "w_down")):
                    module.weight.data.mul_(residual_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.d_model))
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

        if self.config.share_embeddings:
            self.lm_head.weight = self.shared_embedding.weight

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

    def decode_step_cached(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        past_kvs: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        return_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        KV-cached decoding step for fast O(L) autoregressive generation.
        Caches decoder self-attention K/V (with RoPE offset) and static encoder memory cross-attention K/V.
        """
        offset = past_kvs[0][0][0].size(-2) if past_kvs is not None else 0
        x = self.shared_embedding(tgt) * math.sqrt(self.d_model)
        x = self.emb_dropout(x)

        cross_attn_list: List[Optional[torch.Tensor]] = []
        new_kvs: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = []
        for idx, layer in enumerate(self.decoder_layers):
            layer_past = past_kvs[idx] if past_kvs is not None else None
            x, cross_attn, present_kv = layer.forward_cached(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                is_causal=is_causal,
                return_cross_attn=return_cross_attn,
                offset=offset,
                past_kv=layer_past
            )
            new_kvs.append(present_kv)
            if return_cross_attn:
                cross_attn_list.append(cross_attn)

        x = self.decoder_final_norm(x)
        logits = self.lm_head(x)

        if self.config.soft_cap is not None and self.config.soft_cap > 0:
            logits = self.config.soft_cap * torch.tanh(logits / self.config.soft_cap)

        return logits.contiguous(), cross_attn_list, new_kvs

    def _apply_decoding_constraints(
        self,
        logits: torch.Tensor,
        sequences: List[List[int]],
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        script_mask: Optional[torch.Tensor],
        bos_idx: int
    ) -> torch.Tensor:
        """Applies repetition penalty, n-gram blocking, and foreign script masking uniformly to (N, vocab)."""
        out = logits.clone()
        neg_inf = -1e4

        if repetition_penalty != 1.0:
            for b, seq in enumerate(sequences):
                for tok in set(seq):
                    if tok != self.pad_idx and tok != bos_idx:
                        if out[b, tok] > 0:
                            out[b, tok] /= repetition_penalty
                        else:
                            out[b, tok] *= repetition_penalty

        if no_repeat_ngram_size > 0:
            n = no_repeat_ngram_size
            for b, seq in enumerate(sequences):
                if n == 1:
                    for tok in set(seq):
                        if tok != self.pad_idx and tok != bos_idx:
                            out[b, tok] = neg_inf
                elif len(seq) >= n - 1:
                    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i : i + n - 1]) == prefix:
                            out[b, seq[i + n - 1]] = neg_inf

        if script_mask is not None:
            out.masked_fill_(script_mask.to(out.device), neg_inf)

        return out

    @staticmethod
    def _select_past_kvs(
        past_kvs: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        beam_indices: torch.Tensor
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        """Reorders cached self-attention and cross-attention K/V tensors by selected beam indices."""
        selected = []
        for (sk, sv), (ck, cv) in past_kvs:
            selected.append((
                (sk.index_select(0, beam_indices), sv.index_select(0, beam_indices)),
                (ck.index_select(0, beam_indices), cv.index_select(0, beam_indices))
            ))
        return selected

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
    def _beam_search_single_kv(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        init_tokens: List[int],
        max_len: int,
        num_beams: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        length_penalty: float,
        script_mask: Optional[torch.Tensor],
        bos_idx: int,
        eos_idx: int
    ) -> List[Tuple[List[int], float]]:
        """Batched KV-cached beam search for a single source sequence."""
        device = memory.device

        def score_fn(seq_len: int, raw_score: float) -> float:
            lp = ((5.0 + seq_len) / 6.0) ** length_penalty if length_penalty > 0 else 1.0
            return raw_score / lp

        # Prefill step with initial tokens -> batch=1
        prefix_tensor = torch.tensor([init_tokens], dtype=torch.long, device=device)
        logits, _, past_kvs = self.decode_step_cached(
            prefix_tensor, memory=memory, past_kvs=None, memory_mask=src_mask, is_causal=True
        )
        next_logits = self._apply_decoding_constraints(
            logits[:, -1, :], [init_tokens], repetition_penalty, no_repeat_ngram_size, script_mask, bos_idx
        )
        log_probs = F.log_softmax(next_logits[0], dim=-1)
        topk_lp, topk_tok = torch.topk(log_probs, min(num_beams * 2, log_probs.size(-1)))

        active_beams: List[Tuple[List[int], float, int]] = []
        completed: List[Tuple[List[int], float]] = []

        for lp, tok in zip(topk_lp.tolist(), topk_tok.tolist()):
            if lp <= -1e3:
                continue
            new_seq = init_tokens + [tok]
            if tok == eos_idx:
                completed.append((new_seq, lp))
            else:
                active_beams.append((new_seq, lp, 0))

        active_beams.sort(key=lambda c: score_fn(len(c[0]), c[1]), reverse=True)
        active_beams = active_beams[:num_beams]

        if not active_beams:
            pool = completed if completed else [(init_tokens, 0.0)]
            pool.sort(key=lambda c: score_fn(len(c[0]), c[1]), reverse=True)
            return pool[:num_beams]

        # Expand past_kvs for surviving active beams (all share parent index 0)
        parent_idx = torch.zeros(len(active_beams), dtype=torch.long, device=device)
        beam_kvs = self._select_past_kvs(past_kvs, parent_idx)

        for _ in range(1, max_len):
            k_active = len(active_beams)
            last_tokens = torch.tensor([[b[0][-1]] for b in active_beams], dtype=torch.long, device=device)
            b_mem = memory.expand(k_active, -1, -1)
            b_mask = src_mask.expand(k_active, -1, -1)

            logits, _, new_kvs = self.decode_step_cached(
                last_tokens, memory=b_mem, past_kvs=beam_kvs, memory_mask=b_mask, is_causal=True
            )
            constrained = self._apply_decoding_constraints(
                logits[:, -1, :], [b[0] for b in active_beams],
                repetition_penalty, no_repeat_ngram_size, script_mask, bos_idx
            )
            step_log_probs = F.log_softmax(constrained, dim=-1)
            topk_lp, topk_tok = torch.topk(step_log_probs, min(num_beams * 2, step_log_probs.size(-1)), dim=-1)

            candidates: List[Tuple[List[int], float, int]] = []
            for beam_i, (seq, score, _) in enumerate(active_beams):
                for lp, tok in zip(topk_lp[beam_i].tolist(), topk_tok[beam_i].tolist()):
                    if lp <= -1e3:
                        continue
                    new_seq = seq + [tok]
                    new_score = score + lp
                    if tok == eos_idx:
                        completed.append((new_seq, new_score))
                    else:
                        candidates.append((new_seq, new_score, beam_i))

            if len(completed) >= num_beams:
                completed_sorted = sorted(completed, key=lambda c: score_fn(len(c[0]), c[1]), reverse=True)
                worst_top_completed = score_fn(len(completed_sorted[num_beams - 1][0]), completed_sorted[num_beams - 1][1])
                best_active = max(
                    (score_fn(len(c[0]), c[1]) for c in candidates), default=-float("inf")
                )
                if worst_top_completed >= best_active:
                    break

            if not candidates:
                break

            candidates.sort(key=lambda c: score_fn(len(c[0]), c[1]), reverse=True)
            chosen = candidates[:num_beams]
            parent_idx = torch.tensor([c[2] for c in chosen], dtype=torch.long, device=device)
            beam_kvs = self._select_past_kvs(new_kvs, parent_idx)
            active_beams = chosen

        pool = completed if completed else [(b[0], b[1]) for b in active_beams]
        pool.sort(key=lambda c: score_fn(len(c[0]), c[1]), reverse=True)
        return pool[:num_beams]

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
        prefix_tokens: Optional[List[int]] = None,
        bos_idx: Optional[int] = None,
        eos_idx: Optional[int] = None
    ) -> List[List[int]]:
        """
        KV-cached autoregressive sequence generation with Greedy or Batched Beam Search.
        Includes Gemma 2 logit soft-capping, script masking, n-gram repetition blocking,
        and robust finished-sequence clamping for batch inference.
        """
        self.eval()
        batch_size = src.size(0)
        bos = self.bos_idx if bos_idx is None else bos_idx
        eos = self.eos_idx if eos_idx is None else eos_idx

        if src.numel() == 0 or src.size(1) == 0:
            return [[] for _ in range(batch_size)]
        if max_len <= 0:
            return [[bos] for _ in range(batch_size)]

        device = self.shared_embedding.weight.device
        if src.device != device:
            src = src.to(device)

        num_beams = max(1, int(num_beams))
        src_mask = self.make_src_mask(src)
        memory, _ = self.encode(src, src_mask=src_mask, return_attn=False)

        init_tokens = [bos] + list(prefix_tokens) if prefix_tokens else [bos]

        if num_beams == 1:
            # Fast KV-cached Greedy Decoding with batch-correct finished sequence clamping
            generated = torch.tensor([init_tokens] * batch_size, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            past_kvs = None
            step_input = generated

            for _ in range(max_len):
                logits, _, past_kvs = self.decode_step_cached(
                    step_input,
                    memory=memory,
                    past_kvs=past_kvs,
                    memory_mask=src_mask,
                    is_causal=True
                )
                next_token_logits = self._apply_decoding_constraints(
                    logits[:, -1, :],
                    generated.tolist(),
                    repetition_penalty,
                    no_repeat_ngram_size,
                    script_mask,
                    bos
                )
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                next_tokens = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_tokens, self.pad_idx),
                    next_tokens
                )
                finished = finished | (next_tokens.squeeze(-1) == eos)
                generated = torch.cat([generated, next_tokens], dim=-1)
                step_input = next_tokens

                if finished.all():
                    break

            results = []
            for seq in generated.tolist():
                clean_seq = []
                for tok in seq:
                    clean_seq.append(tok)
                    if tok == eos:
                        break
                results.append(clean_seq)
            return results
        else:
            results = []
            for b in range(batch_size):
                beams = self._beam_search_single_kv(
                    memory=memory[b:b+1],
                    src_mask=src_mask[b:b+1],
                    init_tokens=init_tokens,
                    max_len=max_len,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    length_penalty=length_penalty,
                    script_mask=script_mask,
                    bos_idx=bos,
                    eos_idx=eos
                )
                results.append(beams[0][0])
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
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 2,
        disallowed_mask: Optional[torch.Tensor] = None
    ):
        """Polymorphic greedy decoding wrapper matching classical Seq2SeqTransformer contract."""
        seqs = self.generate(
            src=src,
            max_len=max_len,
            num_beams=1,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            script_mask=disallowed_mask,
            prefix_tokens=prefix_tokens,
            bos_idx=sos_idx,
            eos_idx=eos_idx
        )
        device = self.shared_embedding.weight.device
        max_out_len = max((len(s) for s in seqs), default=1)
        out_tensor = torch.full((len(seqs), max_out_len), self.pad_idx, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            if s:
                out_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        if return_attention:
            src_dev = src.to(device)
            src_mask = self.make_src_mask(src_dev)
            memory, _ = self.encode(src_dev, src_mask=src_mask, return_attn=False)
            _, cross_attns = self.decode_step(
                out_tensor, memory=memory, memory_mask=src_mask, is_causal=True, return_cross_attn=True
            )
            last_attn = cross_attns[-1] if cross_attns else None
            return out_tensor, last_attn

        return out_tensor

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        beam_size: int = 3,
        max_len: int = 64,
        length_penalty: float = 0.6,
        sos_idx: int = 2,
        eos_idx: int = 3,
        prefix_tokens: Optional[List[int]] = None,
        disallowed_mask: Optional[torch.Tensor] = None,
        no_repeat_ngram_size: int = 2,
        repetition_penalty: float = 1.15
    ) -> List[Tuple[torch.Tensor, float]]:
        """Polymorphic beam search decoding wrapper returning [(seq_tensor, score)] matching classical signature."""
        self.eval()
        device = self.shared_embedding.weight.device
        src_dev = src.to(device)
        src_mask = self.make_src_mask(src_dev)
        memory, _ = self.encode(src_dev, src_mask=src_mask, return_attn=False)
        init_tokens = [sos_idx] + list(prefix_tokens) if prefix_tokens else [sos_idx]

        beams = self._beam_search_single_kv(
            memory=memory[0:1],
            src_mask=src_mask[0:1],
            init_tokens=init_tokens,
            max_len=max_len,
            num_beams=max(1, int(beam_size)),
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            script_mask=disallowed_mask,
            bos_idx=sos_idx,
            eos_idx=eos_idx
        )
        return [
            (torch.tensor([seq], dtype=torch.long, device=device), float(score))
            for seq, score in beams
        ]
