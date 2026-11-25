#!/usr/bin/env python3
"""
TranslateMe 100M Parameter SOTA Modern Transformer Multi-GPU Trainer
Tailored for Kaggle Dual Nvidia Tesla T4 (2x16GB VRAM) / L4 / A100 Cloud GPUs

100% Self-Contained & State-Dict Compatible with `ml_lab/custom_transformer/modern_transformer.py`:
- Architecture: ~111.7M ModernSeq2SeqTransformer (Pre-LN RMSNorm + RoPE + SwiGLU + QK-Norm + LayerScale + Soft-Cap)
- FlashAttention SDPA: Pure causal SDPA without materialized O(N^2) causal mask overhead
- Pre-Tokenized Rust BPE Cache + LengthBucketBatchSampler: Cuts padding waste from ~65% to <10% (2-3x faster!)
- Optimization: Fused AdamW (with 1D norm/gamma/bias decay exemption), CosineWarmup or Noam LR, Z-loss, BF16/FP16 AMP
- Persistence: Exact deterministic mid-epoch resume, atomic saves, lightweight best export, disk-budget cleanup
- Verification: KV-cached live translation probes after every epoch
"""

import os
import sys
import math
import time
import json
import signal
import random
import hashlib
import unicodedata
import re
import urllib.request
import ssl
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pyarrow.parquet as pq
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pq = None
    SSL_CTX = ssl.create_default_context()

try:
    from tokenizers import Tokenizer, decoders, pre_tokenizers, normalizers
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
except ImportError:
    Tokenizer = None

# ==============================================================================
# 1. 100% FROM-SCRATCH MULTILINGUAL SUBWORD TOKENIZER (16,384 VOCAB)
# ==============================================================================

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<2en>", "<2hi>", "<2ta>", "<2te>"]


def get_control_token(lang: str) -> str:
    return f"<2{lang.strip().lower()}>"


class MultilingualSubwordTokenizer:
    """
    Multilingual subword tokenizer wrapper using Byte-Pair Encoding (BPE).
    Trains vocabulary from scratch on multilingual text without pretrained weights.
    """
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<bos>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, tokenizer_file: Optional[str] = None):
        if tokenizer_file and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            self.tokenizer.decoder = decoders.ByteLevel()
        else:
            self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
            self.tokenizer.normalizer = normalizers.NFC()
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

        self._cache_special_token_ids()

    def _cache_special_token_ids(self):
        try:
            self.pad_id = self.tokenizer.token_to_id("<pad>") if self.tokenizer.token_to_id("<pad>") is not None else 0
            self.unk_id = self.tokenizer.token_to_id("<unk>") if self.tokenizer.token_to_id("<unk>") is not None else 1
            bos_cand = self.tokenizer.token_to_id("<bos>")
            self.sos_id = bos_cand if bos_cand is not None else (self.tokenizer.token_to_id("<sos>") or 2)
            self.bos_id = self.sos_id
            self.eos_id = self.tokenizer.token_to_id("<eos>") if self.tokenizer.token_to_id("<eos>") is not None else 3
        except Exception:
            self.pad_id = 0
            self.unk_id = 1
            self.sos_id = 2
            self.bos_id = 2
            self.eos_id = 3

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_control_token_id(self, target_lang: str) -> int:
        token_str = get_control_token(target_lang)
        tid = self.tokenizer.token_to_id(token_str)
        if tid is None:
            raise ValueError(f"Control token '{token_str}' not in vocabulary.")
        return tid

    def train_from_iterator(self, iterator: Iterable[str], vocab_size: int = 16384, min_frequency: int = 2):
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._cache_special_token_ids()

    def encode(self, text: str, add_special_tokens: bool = True, target_lang: Optional[str] = None) -> List[int]:
        enc = self.tokenizer.encode(text.strip())
        tokens = enc.ids
        if add_special_tokens:
            prefix = [self.bos_id]
            if target_lang is not None:
                prefix.append(self.get_control_token_id(target_lang))
            tokens = prefix + tokens + [self.eos_id]
        return tokens

    def encode_batch_raw(self, texts: List[str]) -> List[List[int]]:
        """Multi-threaded Rust batch encoding for blazing-fast dataset pre-tokenization."""
        return [enc.ids for enc in self.tokenizer.encode_batch(texts)]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            filtered = [
                i for i in token_ids
                if i not in (self.pad_id, self.bos_id, self.eos_id, self.unk_id)
                and not (self.tokenizer.id_to_token(i) or "").startswith("<2")
            ]
            return self.tokenizer.decode(filtered)
        return self.tokenizer.decode(token_ids)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.tokenizer.save(filepath)

    @classmethod
    def load(cls, filepath: str) -> "MultilingualSubwordTokenizer":
        return cls(tokenizer_file=filepath)


# ==============================================================================
# 2. DATASET INGESTION: AI4BHARAT SAMANANTAR PARQUET STREAMER
# ==============================================================================

SAMANANTAR_SHARDS = {
    "hi": ["https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/hi/train-00000-of-00008.parquet"],
    "ta": ["https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/ta/train-00000-of-00004.parquet"],
    "te": ["https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/te/train-00000-of-00003.parquet"]
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\r\t\f\v]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_pair(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    if len(src) < 4 or len(tgt) < 4:
        return False
    if len(src) > 300 or len(tgt) > 300:
        return False
    s_words = src.split()
    t_words = tgt.split()
    if len(s_words) < 2 or len(t_words) < 2:
        return False
    if len(s_words) > 100 or len(t_words) > 100:
        return False
    ratio = len(s_words) / max(len(t_words), 1)
    if ratio > 3.0 or ratio < 0.33:
        return False
    if src.strip().lower() == tgt.strip().lower():
        return False
    return True


def fetch_samanantar_shard(url: str, max_pairs: int, cache_dir: Path) -> List[Tuple[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-2] + "_" + url.split("/")[-1]
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 10 * 1024 * 1024:
        print(f"  Found cached shard: {local_path.name} ({local_path.stat().st_size / (1024*1024):.1f} MB)")
        table = pq.read_table(local_path)
    else:
        print(f"  Downloading parquet shard: {url.split('/')[-1]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=180) as resp:
            content = resp.read()
        print(f"  Downloaded {len(content) / (1024*1024):.1f} MB. Caching to {local_path.name}...")
        with open(local_path, "wb") as f:
            f.write(content)
        table = pq.read_table(local_path)

    src_col = table["src"].to_pylist()
    tgt_col = table["tgt"].to_pylist()
    pairs = []
    seen = set()

    for s_raw, t_raw in zip(src_col, tgt_col):
        s = clean_text(s_raw)
        t = clean_text(t_raw)
        if is_valid_pair(s, t):
            h = hashlib.sha256(f"{s.lower()}|||{t.lower()}".encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                pairs.append((s, t))
                if len(pairs) >= max_pairs:
                    break

    print(f"  Extracted {len(pairs):,} clean pairs.")
    return pairs


def build_optimal_dataset(
    target_hi: int = 800000,
    target_ta: int = 400000,
    target_te: int = 400000,
    output_dir: str = "ml_lab/datasets/processed",
    cache_dir: str = "ml_lab/datasets/real_corpus",
    seed: int = 42
):
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    c_path = Path(cache_dir)

    targets = {"hi": target_hi, "ta": target_ta, "te": target_te}
    all_pairs: Dict[str, List[Tuple[str, str]]] = {}

    print("=" * 70)
    print("INGESTING PARALLEL CORPUS (AI4BHARAT SAMANANTAR)")
    print(f"Targets: HI={target_hi:,} | TA={target_ta:,} | TE={target_te:,}")
    print("=" * 70)

    for lang, count in targets.items():
        print(f"\nProcessing English <-> {lang.upper()} ({count:,} pairs)...")
        collected = []
        for url in SAMANANTAR_SHARDS[lang]:
            needed = count - len(collected)
            if needed <= 0:
                break
            pairs = fetch_samanantar_shard(url, needed, c_path)
            collected.extend(pairs)
        all_pairs[lang] = collected

    train_data = []
    val_data = []
    test_data = []

    for lang, pairs in all_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_train = int(0.90 * n)
        n_val = int(0.05 * n)

        train_s = pairs[:n_train]
        val_s = pairs[n_train:n_train + n_val]
        test_s = pairs[n_train + n_val:]

        for s, d_set in [(train_s, train_data), (val_s, val_data), (test_s, test_data)]:
            for en, indic in s:
                d_set.append({"source": en, "target": indic, "target_lang": lang, "source_lang": "en"})
                d_set.append({"source": indic, "target": en, "target_lang": "en", "source_lang": lang})

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print("\nPartitioning Finished (Record Isolation):")
    print(f"  Train: {len(train_data):,} samples | Val: {len(val_data):,} samples | Test: {len(test_data):,} samples")

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data), ("test.jsonl", test_data)]:
        p = out_path / name
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {name} ({p.stat().st_size / (1024*1024):.1f} MB)")


# ==============================================================================
# 3. 100% CUSTOM SOTA MODERN TRANSFORMER (EXACT STATE-DICT PARITY WITH CORE LIB)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight.type_as(output)


def precompute_freqs_cis(dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    seq_len = x.size(-2)
    c = cos[offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    s = sin[offset : offset + seq_len, :].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = precompute_freqs_cis(dim, max_seq_len=max_seq_len, theta=theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        needed_len = offset + max(q.size(-2), k.size(-2))
        if needed_len > self.max_seq_len:
            new_max = max(needed_len, self.max_seq_len * 2)
            new_cos, new_sin = precompute_freqs_cis(self.dim, max_seq_len=new_max, theta=self.theta)
            self.cos = new_cos.to(device=self.cos.device, dtype=self.cos.dtype)
            self.sin = new_sin.to(device=self.sin.device, dtype=self.sin.dtype)
            self.max_seq_len = new_max
        return apply_rotary_emb(q, self.cos, self.sin, offset=offset), apply_rotary_emb(k, self.cos, self.sin, offset=offset)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_ff is None:
            raw_dff = int(2 * (4 * d_model) / 3)
            d_ff = ((raw_dff + 63) // 64) * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(hidden))


class ModernMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_qk_norm: bool = True,
        use_rope: bool = False,
        rope_max_seq_len: int = 4096
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.q_norm = RMSNorm(self.d_k) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_k) if use_qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(dim=self.d_k, max_seq_len=rope_max_seq_len) if use_rope else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward_cached(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        offset: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_cross_attn: bool = False
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, Sq, _ = q.shape
        q_proj = self.w_q(q).view(B, Sq, self.n_heads, self.d_k).transpose(1, 2)
        if self.use_qk_norm:
            q_proj = self.q_norm(q_proj)

        if is_cross_attn and past_kv is not None:
            k_proj, v_proj = past_kv
            if self.use_rope and self.rope is not None:
                q_proj, _ = self.rope(q_proj, q_proj, offset=offset)
        else:
            Sk = k.size(1)
            k_proj = self.w_k(k).view(B, Sk, self.n_heads, self.d_k).transpose(1, 2)
            v_proj = self.w_v(v).view(B, Sk, self.n_heads, self.d_k).transpose(1, 2)
            if self.use_qk_norm:
                k_proj = self.k_norm(k_proj)
            if self.use_rope and self.rope is not None:
                q_proj, k_proj = self.rope(q_proj, k_proj, offset=offset)
            if past_kv is not None and not is_cross_attn:
                k_proj = torch.cat([past_kv[0], k_proj], dim=-2)
                v_proj = torch.cat([past_kv[1], v_proj], dim=-2)

        present_kv = (k_proj, v_proj)
        Sk_total = k_proj.size(-2)
        attn_dropout = self.dropout_p if self.training else 0.0

        sdpa_mask = mask.unsqueeze(1) if (mask is not None and mask.dim() == 3) else mask
        if is_causal:
            if past_kv is not None and Sq == 1 and sdpa_mask is None:
                out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=None, dropout_p=attn_dropout, is_causal=False)
            elif sdpa_mask is None and past_kv is None and Sq == Sk_total:
                out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=None, dropout_p=attn_dropout, is_causal=True)
            else:
                causal_bool = torch.ones((Sq, Sk_total), dtype=torch.bool, device=q.device).tril(diagonal=Sk_total - Sq).unsqueeze(0).unsqueeze(0)
                combined = (sdpa_mask & causal_bool) if (sdpa_mask is not None and sdpa_mask.dtype == torch.bool) else causal_bool
                out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=combined, dropout_p=attn_dropout, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj, attn_mask=sdpa_mask, dropout_p=attn_dropout, is_causal=False)

        out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        return self.w_o(out), present_kv

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> Tuple[torch.Tensor, None]:
        out, _ = self.forward_cached(q, k, v, mask=mask, is_causal=is_causal)
        return out, None


class ModernEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_layerscale: bool = True,
        layerscale_init: float = 0.1
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernMultiHeadAttention(d_model, n_heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rope=use_rope)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.use_layerscale = use_layerscale
        if use_layerscale:
            self.gamma_1 = nn.Parameter(torch.ones(d_model) * layerscale_init)
            self.gamma_2 = nn.Parameter(torch.ones(d_model) * layerscale_init)
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, mask=mask, is_causal=False)
        x = x + self.dropout(self.gamma_1 * attn_out)
        x = x + self.dropout(self.gamma_2 * self.ffn(self.norm2(x)))
        return x


class ModernDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_layerscale: bool = True,
        layerscale_init: float = 0.1
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernMultiHeadAttention(d_model, n_heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rope=use_rope)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = ModernMultiHeadAttention(d_model, n_heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rope=False)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
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

    def forward_cached(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        offset: int = 0,
        past_kv: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        self_past, cross_past = past_kv if past_kv is not None else (None, None)

        norm_x = self.norm1(x)
        self_out, new_self = self.self_attn.forward_cached(
            norm_x, norm_x, norm_x, mask=tgt_mask, is_causal=is_causal, offset=offset, past_kv=self_past, is_cross_attn=False
        )
        x = x + self.dropout(self.gamma_1 * self_out)

        norm_x = self.norm2(x)
        cross_out, new_cross = self.cross_attn.forward_cached(
            norm_x, memory, memory, mask=memory_mask, is_causal=False, offset=0, past_kv=cross_past, is_cross_attn=True
        )
        x = x + self.dropout(self.gamma_2 * cross_out)

        x = x + self.dropout(self.gamma_3 * self.ffn(self.norm3(x)))
        return x, (new_self, new_cross)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        out, _ = self.forward_cached(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, is_causal=is_causal)
        return out


class ModernSeq2SeqTransformer(nn.Module):
    """
    111.7M Parameter SOTA Multilingual Transformer.
    Identical module tree and parameter naming to `ml_lab/custom_transformer/modern_transformer.py`
    so checkpoints load directly into the backend inference service with strict=True.
    """
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 768,
        n_heads: int = 12,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        share_embeddings: bool = True,
        use_layerscale: bool = True,
        layerscale_init: float = 0.1,
        soft_cap: float = 30.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.share_embeddings = share_embeddings
        self.use_layerscale = use_layerscale
        self.layerscale_init = layerscale_init
        self.soft_cap = soft_cap

        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.encoder_layers = nn.ModuleList([
            ModernEncoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                use_qk_norm=use_qk_norm, use_rope=use_rope,
                use_layerscale=use_layerscale, layerscale_init=layerscale_init
            )
            for _ in range(num_encoder_layers)
        ])
        self.encoder_final_norm = RMSNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            ModernDecoderLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
                use_qk_norm=use_qk_norm, use_rope=use_rope,
                use_layerscale=use_layerscale, layerscale_init=layerscale_init
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if share_embeddings:
            self.lm_head.weight = self.shared_embedding.weight

        self._init_weights()

    def _init_weights(self):
        total_depth = max(1, self.num_encoder_layers + self.num_decoder_layers)
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
        if self.share_embeddings:
            self.lm_head.weight = self.shared_embedding.weight

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != self.pad_idx).unsqueeze(1)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        x = self.emb_dropout(self.shared_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return self.encoder_final_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask=src_mask)

        x = self.emb_dropout(self.shared_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            # tgt_mask=None + is_causal=True triggers pure FlashAttention / Memory-Efficient SDPA
            x = layer(x, memory=memory, tgt_mask=None, memory_mask=src_mask, is_causal=True)

        x = self.decoder_final_norm(x)
        logits = self.lm_head(x)
        if self.soft_cap and self.soft_cap > 0:
            logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
        return logits.contiguous()

    @torch.no_grad()
    def greedy_generate(self, src: torch.Tensor, prefix_tokens: Optional[List[int]] = None, max_len: int = 64) -> List[int]:
        """Fast KV-cached greedy generator for live training probes."""
        self.eval()
        device = self.shared_embedding.weight.device
        src = src.to(device)
        src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask=src_mask)

        init_tokens = [self.bos_idx] + (prefix_tokens or [])
        generated = list(init_tokens)
        step_input = torch.tensor([init_tokens], dtype=torch.long, device=device)
        past_kvs = None

        for _ in range(max_len):
            offset = past_kvs[0][0][0].size(-2) if past_kvs is not None else 0
            x = self.shared_embedding(step_input) * math.sqrt(self.d_model)
            new_kvs = []
            for idx, layer in enumerate(self.decoder_layers):
                x, kv = layer.forward_cached(
                    x, memory=memory, tgt_mask=None, memory_mask=src_mask,
                    is_causal=True, offset=offset, past_kv=(past_kvs[idx] if past_kvs else None)
                )
                new_kvs.append(kv)
            past_kvs = new_kvs

            logits = self.lm_head(self.decoder_final_norm(x))[:, -1, :]
            if self.soft_cap and self.soft_cap > 0:
                logits = self.soft_cap * torch.tanh(logits / self.soft_cap)
            next_tok = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_tok)
            if next_tok == self.eos_idx:
                break
            step_input = torch.tensor([[next_tok]], dtype=torch.long, device=device)

        self.train()
        return generated


# ==============================================================================
# 4. PRE-TOKENIZED DATASET + LENGTH-BUCKETED SAMPLER + SCHEDULERS
# ==============================================================================

def get_preferred_amp_dtype(device: torch.device) -> Tuple[torch.dtype, bool]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.float32, False
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


class CosineWarmupScheduler:
    """Linear warmup + Cosine annealing LR decay down to min_lr_ratio * peak_lr."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, peak_lr: float = 5e-4, min_lr_ratio: float = 0.05):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.peak_lr = peak_lr
        self.min_lr_ratio = min_lr_ratio
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.peak_lr * (self.step_num / self.warmup_steps)
        else:
            progress = min(1.0, (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.peak_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "min_lr_ratio": self.min_lr_ratio
        }

    def load_state_dict(self, state):
        self.step_num = state["step_num"]


class NoamScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int = 768, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_num": self.step_num, "d_model": self.d_model, "warmup_steps": self.warmup_steps, "factor": self.factor}

    def load_state_dict(self, state):
        self.step_num = state["step_num"]


class PreTokenizedNMTDataset(Dataset):
    """
    Loads parallel JSONL once, batch-tokenizes via Rust HuggingFace tokenizer (~10s for 2.88M rows),
    caches compact packed int16/int32 tensors to disk, and serves zero-CPU-overhead samples.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: MultilingualSubwordTokenizer,
        max_len: int = 128,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.max_len = max_len
        self.tok_dtype = torch.int16 if tokenizer.vocab_size <= 32767 else torch.int32

        if cache_path and os.path.exists(cache_path) and max_samples is None:
            print(f"  [CACHE HIT] Loading pre-tokenized dataset from {Path(cache_path).name}...")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.src_data = cached["src_data"]
            self.src_offsets = cached["src_offsets"]
            self.tgt_data = cached["tgt_data"]
            self.tgt_offsets = cached["tgt_offsets"]
            self.lengths = cached["lengths"]
            return

        t0 = time.time()
        print(f"  [PRE-TOKENIZE] Reading & batch-tokenizing {Path(jsonl_path).name}...")
        sources: List[str] = []
        targets: List[str] = []
        ctrl_ids: List[int] = []
        ctrl_cache = {lang: tokenizer.get_control_token_id(lang) for lang in ["en", "hi", "ta", "te"]}

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                d = json.loads(line)
                sources.append(d["source"].strip())
                targets.append(d["target"].strip())
                tlang = d["target_lang"]
                if tlang not in ctrl_cache:
                    ctrl_cache[tlang] = tokenizer.get_control_token_id(tlang)
                ctrl_ids.append(ctrl_cache[tlang])

        n = len(sources)
        src_flat: List[int] = []
        tgt_flat: List[int] = []
        src_off = [0]
        tgt_off = [0]
        lengths: List[int] = []

        chunk_size = 50000
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            src_enc = tokenizer.encode_batch_raw(sources[start:end])
            tgt_enc = tokenizer.encode_batch_raw(targets[start:end])

            for i, (s_ids, t_ids) in enumerate(zip(src_enc, tgt_enc)):
                s_seq = [self.bos_id] + s_ids + [self.eos_id]
                if len(s_seq) > max_len:
                    s_seq = s_seq[:max_len - 1] + [self.eos_id]

                c_id = ctrl_ids[start + i]
                t_seq = [self.bos_id, c_id] + t_ids + [self.eos_id]
                if len(t_seq) > max_len:
                    t_seq = t_seq[:max_len - 1] + [self.eos_id]

                src_flat.extend(s_seq)
                tgt_flat.extend(t_seq)
                src_off.append(len(src_flat))
                tgt_off.append(len(tgt_flat))
                lengths.append(len(s_seq) + len(t_seq))

        self.src_data = torch.tensor(src_flat, dtype=self.tok_dtype)
        self.src_offsets = torch.tensor(src_off, dtype=torch.int64)
        self.tgt_data = torch.tensor(tgt_flat, dtype=self.tok_dtype)
        self.tgt_offsets = torch.tensor(tgt_off, dtype=torch.int64)
        self.lengths = lengths

        print(f"  [PRE-TOKENIZE] Tokenized {n:,} pairs in {time.time() - t0:.1f}s.")
        if cache_path and max_samples is None:
            tmp_cache = f"{cache_path}.tmp"
            torch.save({
                "src_data": self.src_data,
                "src_offsets": self.src_offsets,
                "tgt_data": self.tgt_data,
                "tgt_offsets": self.tgt_offsets,
                "lengths": self.lengths,
            }, tmp_cache)
            os.replace(tmp_cache, cache_path)
            print(f"  [CACHE SAVED] Stored compact tensor cache: {Path(cache_path).name}")

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s0, s1 = int(self.src_offsets[idx]), int(self.src_offsets[idx + 1])
        t0, t1 = int(self.tgt_offsets[idx]), int(self.tgt_offsets[idx + 1])
        return self.src_data[s0:s1].long(), self.tgt_data[t0:t1].long()


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Groups similar-length pairs into mini-batches to minimize padding waste (2-3x speedup)."""
    def __init__(self, lengths: List[int], batch_size: int, seed: int = 42, mega_batch_mult: int = 40):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.mega_batch_mult = mega_batch_mult
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=g).tolist()

        mega_size = self.batch_size * self.mega_batch_mult
        batches: List[List[int]] = []
        for i in range(0, len(indices), mega_size):
            chunk = indices[i:i + mega_size]
            chunk.sort(key=lambda idx: self.lengths[idx])
            for j in range(0, len(chunk), self.batch_size):
                batches.append(chunk[j:j + self.batch_size])

        batch_perm = torch.randperm(len(batches), generator=g).tolist()
        for b_idx in batch_perm:
            yield batches[b_idx]


def build_tensor_collate_fn(pad_id: int = 0):
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_list = [b[0] for b in batch]
        tgt_in_list = [b[1][:-1] for b in batch]
        tgt_out_list = [b[1][1:] for b in batch]
        return {
            "src": nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id),
            "tgt_in": nn.utils.rnn.pad_sequence(tgt_in_list, batch_first=True, padding_value=pad_id),
            "tgt_out": nn.utils.rnn.pad_sequence(tgt_out_list, batch_first=True, padding_value=pad_id),
        }
    return collate_fn


def configure_adamw_optimizer(
    model: nn.Module,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.98),
    eps: float = 1e-8
) -> torch.optim.AdamW:
    """Applies weight decay to 2D matrices only; exempts 1D norms, biases, and LayerScale gammas."""
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name.lower() or "bias" in name.lower() or "gamma" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    use_fused = (
        decay_params
        and decay_params[0].device.type == "cuda"
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )
    kwargs: Dict[str, Any] = {"lr": lr, "betas": betas, "eps": eps}
    if use_fused:
        kwargs["fused"] = True
    return torch.optim.AdamW(optim_groups, **kwargs)


# ==============================================================================
# 5. MULTI-GPU RESUMABLE TRAINER WITH REAL-TIME ETA & DIAGNOSTICS
# ==============================================================================

def format_duration(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "calculating..."
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    return f"{m}m {sec:02d}s"


class Kaggle100MTrainer:
    def __init__(
        self,
        model: ModernSeq2SeqTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str = "ml_lab/checkpoints",
        tokenizer_path: str = "ml_lab/tokenizer/multilingual_tokenizer.json",
        gradient_accumulation_steps: int = 2,
        use_amp: bool = True,
        z_loss_weight: float = 1e-4,
        device: Optional[torch.device] = None,
        save_every_steps: int = 500,
        eval_every_steps: int = 2500
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = tokenizer_path
        self.grad_accum = gradient_accumulation_steps
        self.z_loss_weight = z_loss_weight
        self.save_every_steps = save_every_steps
        self.eval_every_steps = eval_every_steps

        self.amp_dtype, needs_scaler = get_preferred_amp_dtype(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")

        self.is_multi_gpu = torch.cuda.device_count() > 1
        if self.is_multi_gpu:
            print(f"[*] Detected {torch.cuda.device_count()} CUDA GPUs! Enabling torch.nn.DataParallel.")
            self.model_engine = nn.DataParallel(self.model)
        else:
            self.model_engine = self.model

        self.scaler = GradScaler("cuda", enabled=self.use_amp and needs_scaler)
        self.sum_criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx, label_smoothing=0.1, reduction="sum")

        self.epoch = 0
        self.epoch_batch_idx = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self._pause_requested = False

        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}. Triggering graceful checkpoint save...")
        self._pause_requested = True

    def _build_config_dict(self, raw_model: ModernSeq2SeqTransformer) -> Dict[str, Any]:
        return {
            "vocab_size": raw_model.vocab_size,
            "d_model": raw_model.d_model,
            "n_heads": raw_model.n_heads,
            "num_encoder_layers": raw_model.num_encoder_layers,
            "num_decoder_layers": raw_model.num_decoder_layers,
            "d_ff": raw_model.d_ff,
            "dropout": raw_model.dropout_rate,
            "use_rope": raw_model.use_rope,
            "use_qk_norm": raw_model.use_qk_norm,
            "share_embeddings": raw_model.share_embeddings,
            "use_layerscale": raw_model.use_layerscale,
            "layerscale_init": raw_model.layerscale_init,
            "soft_cap": raw_model.soft_cap,
            "model_type": "modern",
        }

    def save_atomic_checkpoint(self, filename: str, is_best: bool = False):
        save_path = self.checkpoint_dir / filename
        tmp_path = self.checkpoint_dir / f"{filename}.tmp"

        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        cfg_dict = self._build_config_dict(raw_model)

        state = {
            "epoch": self.epoch,
            "epoch_batch_idx": self.epoch_batch_idx,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "config": cfg_dict,
            "tokenizer_path": self.tokenizer_path,
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "rng_state": {
                "cpu": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state() if np is not None else None,
            },
            "timestamp": time.time()
        }
        torch.save(state, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"[CHECKPOINT] Saved {save_path} (Step {self.global_step}, Epoch {self.epoch+1}, Batch {self.epoch_batch_idx})")

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            tmp_best = self.checkpoint_dir / "checkpoint_best.pt.tmp"
            inference_state = {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "model_state_dict": raw_model.state_dict(),
                "config": cfg_dict,
                "tokenizer_path": self.tokenizer_path,
                "train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "timestamp": time.time(),
            }
            torch.save(inference_state, tmp_best)
            os.replace(tmp_best, best_path)
            print(f"[*] Best inference checkpoint updated: {best_path} (Val Loss: {self.best_val_loss:.4f})")

        # Prune older epoch milestones beyond the latest 2 to stay well under Kaggle's 19.5 GB limit
        epoch_files = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(epoch_files) > 2:
            for old_file in epoch_files[:-2]:
                try:
                    old_file.unlink()
                except Exception:
                    pass

    def write_progress_json(self, payload: Dict[str, Any]):
        try:
            prog_path = self.checkpoint_dir / "training_progress.json"
            tmp_path = self.checkpoint_dir / "training_progress.json.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, prog_path)
        except Exception:
            pass

    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False
        print(f"[RESUME] Loading state from {checkpoint_path}...")
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        raw_model.load_state_dict(chk["model_state_dict"])
        if chk.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(chk["optimizer_state_dict"])
        if self.scheduler and chk.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(chk["scheduler_state_dict"])
        if self.scaler.is_enabled() and chk.get("scaler_state_dict"):
            self.scaler.load_state_dict(chk["scaler_state_dict"])

        self.epoch = chk.get("epoch", 0)
        self.epoch_batch_idx = chk.get("epoch_batch_idx", 0)
        self.global_step = chk.get("global_step", 0)
        self.best_val_loss = chk.get("best_val_loss", float("inf"))
        self.train_loss_history = chk.get("train_loss_history", [])
        self.val_loss_history = chk.get("val_loss_history", [])

        rng = chk.get("rng_state")
        if rng:
            if rng.get("cpu") is not None:
                torch.set_rng_state(rng["cpu"].cpu())
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if np is not None and rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])

        print(f"[RESUME SUCCESS] Resumed at Epoch {self.epoch+1}, Batch {self.epoch_batch_idx}, Step {self.global_step} (Best Val: {self.best_val_loss:.4f})")
        return True

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        total_epochs: int = 5,
        log_interval: int = 50,
        tokenizer: Optional[MultilingualSubwordTokenizer] = None
    ):
        self.model_engine.train()
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(self.epoch)

        num_batches = len(train_loader)
        run_start = time.time()
        running_loss = 0.0
        batches_in_window = 0
        batches_processed_run = 0
        tokens_processed_run = 0
        skip_until = self.epoch_batch_idx
        self.optimizer.zero_grad(set_to_none=True)

        if skip_until > 0:
            print(f"  [MID-EPOCH RESUME] Fast-forwarding batches 0..{skip_until-1} in Epoch {self.epoch+1}...")

        try:
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx < skip_until:
                    continue

                if self._pause_requested or Path("pause.flag").exists() or Path("/kaggle/working/pause.flag").exists():
                    print("\n[PAUSE] Pause trigger detected! Saving checkpoint gracefully...")
                    for pf in [Path("pause.flag"), Path("/kaggle/working/pause.flag")]:
                        if pf.exists():
                            try:
                                pf.unlink()
                            except Exception:
                                pass
                    self.epoch_batch_idx = batch_idx
                    self.save_atomic_checkpoint("checkpoint_latest.pt")
                    print(f"[PAUSED] Saved at Epoch {self.epoch+1}, Batch {batch_idx+1}, Step {self.global_step}.")
                    return {"status": "paused", "epoch": self.epoch, "step": self.global_step}

                src = batch["src"].to(self.device, non_blocking=True)
                tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
                tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

                raw_model = self.model_engine.module if self.is_multi_gpu else self.model
                valid_mask = tgt_out != raw_model.pad_idx
                batch_tokens = int(valid_mask.sum().item())
                tokens_processed_run += batch_tokens

                chunk_start = (batch_idx // self.grad_accum) * self.grad_accum
                actual_chunk_size = min(self.grad_accum, num_batches - chunk_start)

                with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model_engine(src, tgt_in)
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = tgt_out.view(-1)
                    batch_sum_loss = self.sum_criterion(flat_logits, flat_targets)
                    raw_loss = batch_sum_loss / max(batch_tokens, 1)

                    total_loss_step = raw_loss
                    if self.z_loss_weight > 0:
                        log_z = torch.logsumexp(flat_logits.float(), dim=-1)
                        z_loss = (log_z ** 2)[valid_mask.view(-1)].mean()
                        total_loss_step = raw_loss + self.z_loss_weight * z_loss

                    loss = total_loss_step / actual_chunk_size

                self.scaler.scale(loss).backward()
                running_loss += raw_loss.item()
                batches_in_window += 1
                batches_processed_run += 1
                self.epoch_batch_idx = batch_idx + 1

                if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == num_batches:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler:
                        self.scheduler.step()
                    self.global_step += 1

                    if self.eval_every_steps > 0 and val_loader and self.global_step % self.eval_every_steps == 0:
                        mid_val = self.evaluate(val_loader, max_batches=40)
                        self.val_loss_history.append(mid_val)
                        is_best = mid_val < self.best_val_loss
                        if is_best:
                            self.best_val_loss = mid_val
                        self.save_atomic_checkpoint("checkpoint_latest.pt", is_best=is_best)
                        print(f"  [MID-EVAL @ Step {self.global_step}] Val Loss: {mid_val:.4f} | Val PPL: {math.exp(min(mid_val, 20)):.2f} | Best: {self.best_val_loss:.4f}")
                    elif self.global_step % self.save_every_steps == 0:
                        self.save_atomic_checkpoint("checkpoint_latest.pt")

                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                    elapsed = time.time() - run_start
                    batches_done = batch_idx + 1
                    batches_per_sec = batches_processed_run / max(elapsed, 0.001)

                    epoch_batches_left = num_batches - batches_done
                    epoch_time_left = epoch_batches_left / max(batches_per_sec, 0.001)
                    epochs_remaining = max(total_epochs - (self.epoch + 1), 0)
                    total_time_left = (epochs_remaining * num_batches + epoch_batches_left) / max(batches_per_sec, 0.001)
                    tok_per_sec = tokens_processed_run / max(elapsed, 0.001)

                    avg_loss = running_loss / max(batches_in_window, 1)
                    ppl = math.exp(min(avg_loss, 20.0))
                    lr = self.optimizer.param_groups[0]["lr"]
                    pct_epoch = (batches_done / num_batches) * 100.0

                    print(
                        f"Epoch {self.epoch+1:2d}/{total_epochs:2d} | "
                        f"Batch {batches_done:5d}/{num_batches:5d} ({pct_epoch:4.1f}%) | "
                        f"Step {self.global_step:6d} | "
                        f"Loss: {avg_loss:5.3f} | PPL: {ppl:6.2f} | "
                        f"LR: {lr:.2e} | Speed: {tok_per_sec:6.0f} tok/s | "
                        f"Epoch Left: {format_duration(epoch_time_left)} | "
                        f"Total ETA: {format_duration(total_time_left)}"
                    )
                    self.train_loss_history.append(avg_loss)
                    running_loss = 0.0
                    batches_in_window = 0

                    self.write_progress_json({
                        "epoch": self.epoch + 1,
                        "total_epochs": total_epochs,
                        "batch": batches_done,
                        "total_batches": num_batches,
                        "global_step": self.global_step,
                        "percent_epoch": round(pct_epoch, 1),
                        "loss": round(avg_loss, 4),
                        "perplexity": round(ppl, 2),
                        "lr": lr,
                        "tok_per_sec": int(tok_per_sec),
                        "epoch_left": format_duration(epoch_time_left),
                        "total_eta": format_duration(total_time_left),
                        "best_val_loss": round(self.best_val_loss, 4) if self.best_val_loss != float("inf") else None,
                        "timestamp": time.time(),
                    })

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Caught Stop / Interrupt signal at Batch {self.epoch_batch_idx}, Step {self.global_step}!")
            print("Saving atomic checkpoint to preserve exact training state on disk...")
            self.save_atomic_checkpoint("checkpoint_latest.pt")
            print(f"[PAUSED] Saved at Epoch {self.epoch+1}, Batch {self.epoch_batch_idx}, Step {self.global_step}.")
            print("[INFO] You can safely restart your Kaggle session to refresh the 12-hour timer. It will resume right here!")
            return {"status": "interrupted", "epoch": self.epoch, "step": self.global_step}

        if val_loader:
            val_loss = self.evaluate(val_loader)
            self.val_loss_history.append(val_loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.epoch += 1
            self.epoch_batch_idx = 0
            self.save_atomic_checkpoint("checkpoint_latest.pt", is_best=is_best)
            self.save_atomic_checkpoint(f"checkpoint_epoch_{self.epoch}.pt")
            print(f"=== Epoch {self.epoch}/{total_epochs} Complete | Val Loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.2f} | Best Val: {self.best_val_loss:.4f} ===")
        else:
            self.epoch += 1
            self.epoch_batch_idx = 0
            self.save_atomic_checkpoint("checkpoint_latest.pt")

        if tokenizer is not None:
            self.run_diagnostic_probes(tokenizer)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, max_batches: Optional[int] = None) -> float:
        self.model_engine.eval()
        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        eval_criterion = nn.CrossEntropyLoss(ignore_index=raw_model.pad_idx, reduction="sum")

        total_loss = 0.0
        total_tokens = 0

        for idx, batch in enumerate(val_loader):
            if max_batches is not None and idx >= max_batches:
                break
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

            with autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model_engine(src, tgt_in)
                loss = eval_criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

            valid_tokens = (tgt_out != raw_model.pad_idx).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

        self.model_engine.train()
        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def run_diagnostic_probes(self, tokenizer: MultilingualSubwordTokenizer):
        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        probes = [
            ("Hello, how are you today?", "hi"),
            ("Machine learning transforms modern technology.", "ta"),
            ("Knowledge empowers people to build the future.", "te"),
            ("नमस्ते, आप कैसे हैं?", "en"),
        ]
        print("\n[LIVE PROBES] Sample Greedy Translations:")
        for text, tgt_lang in probes:
            src_ids = tokenizer.encode(text, add_special_tokens=True)
            src_t = torch.tensor([src_ids], dtype=torch.long)
            ctrl_id = tokenizer.get_control_token_id(tgt_lang)
            gen_ids = raw_model.greedy_generate(src_t, prefix_tokens=[ctrl_id], max_len=48)
            out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"  [->{tgt_lang}] '{text}' => '{out_text}'")
        print()


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def main(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train 100M Modern Transformer on Kaggle / Multi-GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size per step (split across GPUs; bucketed)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup", type=int, default=2000, help="Warmup optimizer steps")
    parser.add_argument("--scheduler", choices=["cosine", "noam"], default="cosine", help="LR scheduler type")
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint save cadence (steps)")
    parser.add_argument("--eval_every", type=int, default=2500, help="Mid-epoch validation cadence (steps)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_size", choices=["optimal", "medium", "smoke"], default="optimal",
                        help="optimal: 1.6M pairs (3.2M bidirectional), medium: 200k, smoke: 20k")
    if args_list is not None:
        args, _ = parser.parse_known_args(args_list)
    else:
        args, _ = parser.parse_known_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    on_kaggle = os.path.exists("/kaggle/working")
    ckpt_dir = "/kaggle/working/checkpoints" if on_kaggle else "ml_lab/checkpoints"

    print("=" * 80)
    print("TRANSLATEME: 100M SOTA MODERN TRANSFORMER TRAINING RUN")
    print(f"Device count: {torch.cuda.device_count()} CUDA GPU(s) | Checkpoint dir: {ckpt_dir}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB VRAM)")
    print("=" * 80)

    # 1. Dataset Verification
    data_root = Path("ml_lab/datasets/processed")
    train_file = str(data_root / "train.jsonl")
    val_file = str(data_root / "val.jsonl")

    if not os.path.exists(train_file):
        print("\n[Phase 1] Processed dataset not found locally. Running Samanantar ingestion...")
        if args.dataset_size == "optimal":
            build_optimal_dataset(target_hi=800000, target_ta=400000, target_te=400000)
        elif args.dataset_size == "medium":
            build_optimal_dataset(target_hi=100000, target_ta=50000, target_te=50000)
        else:
            build_optimal_dataset(target_hi=5000, target_ta=2500, target_te=2500)

    # 2. Tokenizer (Trained strictly on train split)
    tokenizer_path = "ml_lab/tokenizer/multilingual_tokenizer.json"
    tokenizer = MultilingualSubwordTokenizer()

    if os.path.exists(tokenizer_path):
        print(f"\n[Phase 2] Loading existing subword tokenizer from {tokenizer_path}...")
        tokenizer = MultilingualSubwordTokenizer.load(tokenizer_path)
    else:
        print("\n[Phase 2] Training subword BPE tokenizer (16,384 vocab) strictly on train split...")
        texts = []
        with open(train_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx > 250000:
                    break
                d = json.loads(line)
                texts.append(d["source"])
                texts.append(d["target"])
        tokenizer.train_from_iterator(texts, vocab_size=16384, min_frequency=2)
        tokenizer.save(tokenizer_path)

    print(f"Tokenizer active. Vocab size: {tokenizer.vocab_size:,}")

    # 3. Model Initialization (Exact ModernSeq2SeqTransformer 111.7M Architecture)
    print("\n[Phase 3] Initializing ModernSeq2SeqTransformer (~111.7M parameters) from random scratch weights...")
    model = ModernSeq2SeqTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_heads=12,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        use_qk_norm=True,
        use_rope=True,
        share_embeddings=True,
        use_layerscale=True,
        layerscale_init=0.1,
        soft_cap=30.0
    )
    total_params = model.count_parameters()
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 4. Pre-Tokenized Datasets & Length-Bucketed DataLoaders
    print("\n[Phase 4] Preparing Pre-Tokenized Datasets & LengthBucketBatchSampler...")
    train_dataset = PreTokenizedNMTDataset(
        train_file, tokenizer, max_len=128, cache_path=str(data_root / "train_tok_v1.pt")
    )
    val_dataset = PreTokenizedNMTDataset(
        val_file, tokenizer, max_len=128, cache_path=str(data_root / "val_tok_v1.pt")
    )
    print(f"Train instances: {len(train_dataset):,} | Val instances: {len(val_dataset):,}")

    collate_fn = build_tensor_collate_fn(pad_id=tokenizer.pad_id)
    train_sampler = LengthBucketBatchSampler(train_dataset.lengths, batch_size=args.batch_size, seed=42)
    num_workers = 2 if sys.platform != "win32" else 0

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    # 5. Optimizer & Scheduler
    optimizer = configure_adamw_optimizer(model, lr=args.lr if args.scheduler == "cosine" else 1e-7, weight_decay=0.01)
    if args.scheduler == "cosine":
        scheduler = CosineWarmupScheduler(optimizer, warmup_steps=args.warmup, total_steps=total_steps, peak_lr=args.lr)
    else:
        scheduler = NoamScheduler(optimizer, d_model=768, warmup_steps=args.warmup, factor=1.0)

    print(f"Optimizer: Fused AdamW (1D Norm/Gamma/Bias decay-exempt) | Scheduler: {args.scheduler} (steps={total_steps:,})")

    # 6. Trainer Execution
    trainer = Kaggle100MTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        gradient_accumulation_steps=args.grad_accum,
        use_amp=True,
        save_every_steps=args.save_every,
        eval_every_steps=args.eval_every
    )

    candidates = [
        args.resume if args.resume else "",
        str(Path(ckpt_dir) / "checkpoint_latest.pt"),
        "/kaggle/working/checkpoint_latest.pt",
        "/kaggle/working/checkpoints/checkpoint_latest.pt",
        "ml_lab/checkpoints/checkpoint_latest.pt"
    ]
    for ckpt in candidates:
        if ckpt and os.path.exists(ckpt):
            print(f"\n[AUTO-RESUME] Found checkpoint at {ckpt}! Resuming without losing progress...")
            trainer.resume_from_checkpoint(ckpt)
            break

    print(f"\n[Phase 5] Starting Training for {args.epochs} Epochs...")
    for _ in range(trainer.epoch, args.epochs):
        res = trainer.train_epoch(
            train_loader, val_loader, total_epochs=args.epochs, log_interval=50, tokenizer=tokenizer
        )
        if isinstance(res, dict) and res.get("status") in ("paused", "interrupted"):
            print(f"\n[PAUSED] Training halted cleanly. State is preserved on disk in {ckpt_dir}.")
            break

    print(f"\n[Phase 6] Training Complete! Best Val Loss: {trainer.best_val_loss:.4f}")

    if on_kaggle:
        import shutil
        best_pt = Path(ckpt_dir) / "checkpoint_best.pt"
        if best_pt.exists() and str(best_pt) != "/kaggle/working/checkpoint_best.pt":
            shutil.copy2(best_pt, "/kaggle/working/checkpoint_best.pt")
            print("[KAGGLE EXPORT] Copied best model to /kaggle/working/checkpoint_best.pt")
        if os.path.exists(tokenizer_path):
            shutil.copy2(tokenizer_path, "/kaggle/working/multilingual_tokenizer.json")
            print("[KAGGLE EXPORT] Copied tokenizer to /kaggle/working/multilingual_tokenizer.json")


if __name__ == "__main__":
    main()
