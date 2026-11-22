#!/usr/bin/env python3
"""
TranslateMe 100M Parameter Modern Transformer Multi-GPU Trainer
Tailored for Kaggle Dual Nvidia Tesla T4 (2x16GB VRAM) / Single A100 / Dual GPU Cloud

100% Self-Contained:
- Architecture: 111,723,264 Parameter Modern Transformer (Vaswani + RoPE + SwiGLU + RMSNorm + QK-Norm + Soft-Cap)
- Tokenizer: 16,384 Vocab Multilingual Byte-Level BPE trained strictly on train partition
- Dataset Ingestion: AI4Bharat Samanantar (1.6M clean pairs -> 3.2M bidirectional samples)
- Hardware: Automatic Dual GPU (DataParallel) + PyTorch AMP FP16 + Real-Time ETA & Progress
"""

import os
import sys
import math
import time
import json
import random
import hashlib
import unicodedata
import re
import urllib.request
import ssl
import io
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

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

    print(f"\nPartitioning Finished (Record Isolation):")
    print(f"  Train: {len(train_data):,} samples | Val: {len(val_data):,} samples | Test: {len(test_data):,} samples")

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data), ("test.jsonl", test_data)]:
        p = out_path / name
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {name} ({p.stat().st_size / (1024*1024):.1f} MB)")


# ==============================================================================
# 3. 100% CUSTOM MODERN TRANSFORMER ARCHITECTURE (111.72M PARAMETERS)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(variance + self.eps)
        output = normed.to(input_dtype)
        return output * self.weight.type_as(output)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            self._build_cache(max(offset + seq_len, self.max_seq_len * 2))
        return (
            self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(2),
            self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(2)
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.type_as(x)) + (_rotate_half(x) * sin.type_as(x))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class ModernAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, is_causal: bool = False, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.is_causal = is_causal
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = dropout

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, Sq, _ = q_in.shape
        _, Sk, _ = kv_in.shape

        q = self.q_proj(q_in).view(B, Sq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(B, Sk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, Sk, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope and rope_emb is not None:
            cos, sin = rope_emb
            q = apply_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = apply_rope(k.transpose(1, 2), cos, sin).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal and (attn_mask is None) and (Sq == Sk)
        )

        out = out.transpose(1, 2).contiguous().view(B, Sq, self.d_model)
        return self.out_proj(out)


class ModernEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(d_model, n_heads, dropout=dropout, is_causal=False, use_rope=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, rope_emb: Tuple[torch.Tensor, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), rope_emb=rope_emb, attn_mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ModernDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(d_model, n_heads, dropout=dropout, is_causal=True, use_rope=True)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = ModernAttention(d_model, n_heads, dropout=dropout, is_causal=False, use_rope=False)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        rope_emb: Tuple[torch.Tensor, torch.Tensor],
        causal_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x), rope_emb=rope_emb, attn_mask=causal_mask)
        x = x + self.cross_attn(self.norm2(x), memory, rope_emb=None, attn_mask=memory_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class Modern100MTransformer(nn.Module):
    """
    111.72M Parameter Modern Transformer.
    6 Encoder layers, 6 Decoder layers, d_model=768, n_heads=12, d_ff=2048 (SwiGLU).
    Shared tied embeddings for cross-lingual multilingual alignment.
    """
    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 2,
        eos_idx: int = 3,
        soft_cap: float = 30.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.soft_cap = soft_cap

        self.shared_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.emb_scale = math.sqrt(d_model)

        self.rope = RotaryEmbedding(dim=d_model // n_heads, max_seq_len=2048)

        self.encoder_layers = nn.ModuleList([
            ModernEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.encoder_norm = RMSNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            ModernDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_norm = RMSNorm(d_model)

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.shared_emb.weight

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(S, S, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        B, S = src.shape
        rope_emb = self.rope(src, seq_len=S)
        mask = self.make_padding_mask(src)
        x = self.shared_emb(src) * self.emb_scale
        for layer in self.encoder_layers:
            x = layer(x, rope_emb=rope_emb, mask=mask)
        return self.encoder_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        B, S_tgt = tgt.shape
        memory = self.encode(src)

        rope_emb = self.rope(tgt, seq_len=S_tgt)
        tgt_pad_mask = self.make_padding_mask(tgt)
        causal_mask = self.make_causal_mask(S_tgt, tgt.device)
        combined_tgt_mask = tgt_pad_mask & causal_mask

        src_mask = self.make_padding_mask(src)

        x = self.shared_emb(tgt) * self.emb_scale
        for layer in self.decoder_layers:
            x = layer(x, memory=memory, rope_emb=rope_emb, causal_mask=combined_tgt_mask, memory_mask=src_mask)

        x = self.decoder_norm(x)
        logits = self.output_proj(x)
        if self.soft_cap > 0:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap

        return logits.contiguous()


# ==============================================================================
# 4. NOAM SCHEDULER & PYTORCH DATALOADERS
# ==============================================================================

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


class JsonlNMTDataset(Dataset):
    def __init__(self, file_path: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples and idx >= max_samples:
                    break
                d = json.loads(line)
                self.samples.append((d["source"], d["target"], d["target_lang"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def build_nmt_collate_fn(tokenizer: MultilingualSubwordTokenizer, max_len: int = 128):
    def collate_fn(batch):
        src_batch = []
        tgt_in_batch = []
        tgt_out_batch = []

        for src_text, tgt_text, tgt_lang in batch:
            src_tokens = tokenizer.encode(src_text, add_special_tokens=True)
            tgt_tokens = tokenizer.encode(tgt_text, add_special_tokens=True, target_lang=tgt_lang)

            if len(src_tokens) > max_len:
                src_tokens = src_tokens[:max_len - 1] + [tokenizer.eos_id]
            if len(tgt_tokens) > max_len:
                tgt_tokens = tgt_tokens[:max_len - 1] + [tokenizer.eos_id]

            src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
            tgt_in_batch.append(torch.tensor(tgt_tokens[:-1], dtype=torch.long))
            tgt_out_batch.append(torch.tensor(tgt_tokens[1:], dtype=torch.long))

        pad_id = tokenizer.pad_id
        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
        tgt_in_padded = nn.utils.rnn.pad_sequence(tgt_in_batch, batch_first=True, padding_value=pad_id)
        tgt_out_padded = nn.utils.rnn.pad_sequence(tgt_out_batch, batch_first=True, padding_value=pad_id)

        return {"src": src_padded, "tgt_in": tgt_in_padded, "tgt_out": tgt_out_padded}
    return collate_fn


# ==============================================================================
# 5. MULTI-GPU RESUMABLE TRAINER WITH REAL-TIME ETA
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
        model: Modern100MTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_dir: str = "ml_lab/checkpoints",
        gradient_accumulation_steps: int = 2,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
        save_every_steps: int = 1000
    ):
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum = gradient_accumulation_steps
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.save_every_steps = save_every_steps

        # Multi-GPU DataParallel (e.g. 2x Tesla T4 on Kaggle)
        self.is_multi_gpu = torch.cuda.device_count() > 1
        if self.is_multi_gpu:
            print(f"[*] Detected {torch.cuda.device_count()} CUDA GPUs! Enabling torch.nn.DataParallel.")
            self.model_engine = nn.DataParallel(self.model)
        else:
            self.model_engine = self.model

        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.sum_criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx, label_smoothing=0.1, reduction="sum")

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_loss_history = []
        self.val_loss_history = []

    def save_atomic_checkpoint(self, filename: str, is_best: bool = False):
        save_path = self.checkpoint_dir / filename
        tmp_path = self.checkpoint_dir / f"{filename}.tmp"

        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "config": {
                "vocab_size": raw_model.vocab_size,
                "d_model": raw_model.d_model,
                "n_heads": 12,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "d_ff": 2048,
                "use_rope": True,
                "use_qk_norm": True,
                "share_embeddings": True,
                "soft_cap": 30.0,
                "model_type": "modern"
            },
            "timestamp": time.time()
        }
        torch.save(state, tmp_path)
        os.replace(tmp_path, save_path)
        print(f"[CHECKPOINT] Saved atomic checkpoint: {save_path.name} (Step {self.global_step})")

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            tmp_best = self.checkpoint_dir / "checkpoint_best.pt.tmp"
            torch.save(state, tmp_best)
            os.replace(tmp_best, best_path)
            print(f"[*] Best checkpoint updated: checkpoint_best.pt (Val Loss: {self.best_val_loss:.4f})")

    def resume_from_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            return False
        print(f"[RESUME] Loading weights from {checkpoint_path}...")
        chk = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        raw_model.load_state_dict(chk["model_state_dict"])
        self.optimizer.load_state_dict(chk["optimizer_state_dict"])
        if self.scheduler and chk.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(chk["scheduler_state_dict"])
        if self.use_amp and chk.get("scaler_state_dict"):
            self.scaler.load_state_dict(chk["scaler_state_dict"])
        self.epoch = chk.get("epoch", 0)
        self.global_step = chk.get("global_step", 0)
        self.best_val_loss = chk.get("best_val_loss", float("inf"))
        print(f"[RESUME SUCCESS] Resumed at Epoch {self.epoch+1}, Step {self.global_step} (Best Val: {self.best_val_loss:.4f})")
        return True

    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader], total_epochs: int = 5, log_interval: int = 50):
        self.model_engine.train()
        num_batches = len(train_loader)
        epoch_start = time.time()
        running_loss = 0.0
        total_tokens = 0
        self.optimizer.zero_grad()

        # Mid-epoch resume: skip batches that were already processed in this epoch
        batches_done_in_epoch = self.global_step % num_batches if (self.global_step > 0 and self.epoch < total_epochs) else 0
        if batches_done_in_epoch > 0:
            print(f"  [MID-EPOCH RESUME] Fast-forwarding: skipping already trained batches 0 to {batches_done_in_epoch}...")

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < batches_done_in_epoch:
                continue
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

            raw_model = self.model_engine.module if self.is_multi_gpu else self.model
            batch_tokens = (tgt_out != raw_model.pad_idx).sum().item()
            total_tokens += batch_tokens

            chunk_start = (batch_idx // self.grad_accum) * self.grad_accum
            actual_chunk_size = min(self.grad_accum, num_batches - chunk_start)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model_engine(src, tgt_in)
                batch_sum_loss = self.sum_criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
                raw_loss = batch_sum_loss / max(batch_tokens, 1)
                loss = raw_loss / actual_chunk_size

            self.scaler.scale(loss).backward()
            running_loss += raw_loss.item()

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_engine.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.save_every_steps == 0:
                    self.save_atomic_checkpoint("checkpoint_latest.pt")

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                batches_done = batch_idx + 1
                batches_per_sec = batches_done / max(elapsed, 0.001)

                epoch_time_left = (num_batches - batches_done) / max(batches_per_sec, 0.001)
                epochs_remaining = max(total_epochs - (self.epoch + 1), 0)
                total_time_left = (epochs_remaining * num_batches + (num_batches - batches_done)) / max(batches_per_sec, 0.001)
                tok_per_sec = total_tokens / max(elapsed, 0.001)

                chunk_len = log_interval if (batch_idx + 1) % log_interval == 0 else (batch_idx % log_interval + 1)
                avg_loss = running_loss / chunk_len
                ppl = math.exp(min(avg_loss, 20.0))
                lr = self.optimizer.param_groups[0]["lr"]

                pct_epoch = (batches_done / num_batches) * 100.0
                total_pct = ((self.epoch * num_batches + batches_done) / (total_epochs * num_batches)) * 100.0

                print(
                    f"Epoch {self.epoch+1:2d}/{total_epochs:2d} | "
                    f"Batch {batches_done:5d}/{num_batches:5d} ({pct_epoch:4.1f}%) | "
                    f"Loss: {avg_loss:5.3f} | PPL: {ppl:6.2f} | "
                    f"LR: {lr:.2e} | Speed: {tok_per_sec:5.0f} tok/s | "
                    f"Epoch Left: {format_duration(epoch_time_left)} | "
                    f"Total ETA: {format_duration(total_time_left)}"
                )
                self.train_loss_history.append(avg_loss)
                running_loss = 0.0

        if val_loader:
            val_loss = self.evaluate(val_loader)
            self.val_loss_history.append(val_loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_atomic_checkpoint("checkpoint_latest.pt", is_best=is_best)
            self.save_atomic_checkpoint(f"checkpoint_epoch_{self.epoch+1}.pt")
            print(f"=== Epoch {self.epoch+1} Complete | Val Loss: {val_loss:.4f} | Best Val: {self.best_val_loss:.4f} ===")
        else:
            self.save_atomic_checkpoint("checkpoint_latest.pt")

        self.epoch += 1

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        self.model_engine.eval()
        raw_model = self.model_engine.module if self.is_multi_gpu else self.model
        eval_criterion = nn.CrossEntropyLoss(ignore_index=raw_model.pad_idx, reduction="sum")

        total_loss = 0.0
        total_tokens = 0

        for batch in val_loader:
            src = batch["src"].to(self.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(self.device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                logits = self.model_engine(src, tgt_in)
                loss = eval_criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))

            valid_tokens = (tgt_out != raw_model.pad_idx).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

        self.model_engine.train()
        return total_loss / max(total_tokens, 1)


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def main(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Train 100M Modern Transformer on Kaggle / Multi-GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per step (split across GPUs)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--warmup", type=int, default=4000, help="Noam warmup steps")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_size", choices=["optimal", "medium", "smoke"], default="optimal",
                        help="optimal: 1.6M pairs (3.2M bidirectional), medium: 200k, smoke: 20k")
    if args_list is not None:
        args, _ = parser.parse_known_args(args_list)
    else:
        args, _ = parser.parse_known_args()

    print("=" * 80)
    print("TRANSLATEME: 100M PARAMETER MODERN TRANSFORMER TRAINING RUN")
    print(f"Device count: {torch.cuda.device_count()} CUDA GPU(s) available")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GB VRAM)")
    print("=" * 80)

    # 1. Dataset Verification
    train_file = "ml_lab/datasets/processed/train.jsonl"
    val_file = "ml_lab/datasets/processed/val.jsonl"
    test_file = "ml_lab/datasets/processed/test.jsonl"

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
        print(f"\n[Phase 2] Training subword BPE tokenizer (16,384 vocab) strictly on train split...")
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

    # 3. Model Initialization (111.72M Parameters)
    print("\n[Phase 3] Initializing ModernSeq2SeqTransformer (111.72M parameters) from random scratch weights...")
    model = Modern100MTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_heads=12,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        soft_cap=30.0
    )
    total_params = model.count_parameters()
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 4. DataLoaders
    print("\n[Phase 4] Building PyTorch DataLoaders...")
    train_dataset = JsonlNMTDataset(train_file)
    val_dataset = JsonlNMTDataset(val_file)
    print(f"Train instances: {len(train_dataset):,} | Val instances: {len(val_dataset):,}")

    collate_fn = build_nmt_collate_fn(tokenizer, max_len=128)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available()
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = NoamScheduler(optimizer, d_model=768, warmup_steps=args.warmup, factor=1.0)

    # 6. Trainer Execution
    trainer = Kaggle100MTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir="ml_lab/checkpoints",
        gradient_accumulation_steps=args.grad_accum,
        use_amp=True,
        save_every_steps=1000
    )

    # 6. Auto-Resume Check: explicit flag or latest checkpoint on disk
    latest_ckpt = Path(trainer.checkpoint_dir) / "checkpoint_latest.pt"
    if args.resume and os.path.exists(args.resume):
        trainer.resume_from_checkpoint(args.resume)
    elif latest_ckpt.exists():
        print(f"\n[AUTO-RESUME] Found existing checkpoint at {latest_ckpt}! Resuming without losing progress...")
        trainer.resume_from_checkpoint(str(latest_ckpt))

    print(f"\n[Phase 5] Starting Training for {args.epochs} Epochs...")
    for epoch in range(trainer.epoch, args.epochs):
        trainer.train_epoch(train_loader, val_loader, total_epochs=args.epochs, log_interval=50)

    print("\n[Phase 6] Training Complete! Best Val Loss: {:.4f}".format(trainer.best_val_loss))

    if os.path.exists("/kaggle/working"):
        import shutil
        best_pt = "ml_lab/checkpoints/checkpoint_best.pt"
        if os.path.exists(best_pt):
            shutil.copy(best_pt, "/kaggle/working/checkpoint_best.pt")
            print("[KAGGLE EXPORT] Copied best model to /kaggle/working/checkpoint_best.pt")


if __name__ == "__main__":
    main()
