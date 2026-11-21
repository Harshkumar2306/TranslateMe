"""
TranslateMe: Full Production Multilingual Neural Machine Translation
100% Custom PyTorch Transformer Architecture (Vaswani et al., 2017)
Trained strictly from scratch on parallel corpora with 8,192 BPE subwords.
Universal direct translation across English (en), Hindi (hi), Tamil (ta), Telugu (te).
"""

import os
import sys
import math
import time
import json
import gzip
import random
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. 100% FROM-SCRATCH CUSTOM PYTORCH TRANSFORMER ARCHITECTURE
# Section 3 of 'Attention Is All You Need' (Vaswani et al., 2017)
# ==============================================================================

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization layer:
    LN(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """
    A residual connection followed by layer normalization.
    Supports Pre-LN: x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, size: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.pre_ln = pre_ln

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.pre_ln:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000, dropout: float = 0.1, padding_idx: int = 0, shared_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.d_model = d_model
        if shared_embedding is not None:
            self.lut = shared_embedding
        else:
            self.lut = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok_emb = self.lut(x) * math.sqrt(self.d_model)
        return self.pos_encoding(tok_emb)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        output = torch.matmul(attn_weights_dropped, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = q.size(0)
        q_proj = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_proj = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_proj = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attn_out, attn_weights = self.attention(q_proj, k_proj, v_proj, mask=mask)
        concat = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(concat)
        if return_attn:
            return output, attn_weights
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln)
        ])
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        if return_attn:
            norm_x = self.sublayer_connections[0].norm(x) if self.sublayer_connections[0].pre_ln else x
            attn_out, attn_weights = self.self_attn(norm_x, norm_x, norm_x, mask=mask, return_attn=True)
            if self.sublayer_connections[0].pre_ln:
                x = x + self.sublayer_connections[0].dropout(attn_out)
            else:
                x = self.sublayer_connections[0].norm(x + self.sublayer_connections[0].dropout(attn_out))
            x = self.sublayer_connections[1](x, self.feed_forward)
            return x, attn_weights
        else:
            x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
            x = self.sublayer_connections[1](x, self.feed_forward)
            return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1, pre_ln: bool = False, embedding: Optional[TransformerEmbedding] = None):
        super().__init__()
        self.embedding = embedding if embedding is not None else TransformerEmbedding(vocab_size, d_model, max_len=max_len, dropout=dropout)
        self.layers = clones(EncoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, return_all_attentions: bool = False):
        x = self.embedding(src)
        attentions = []
        for layer in self.layers:
            if return_all_attentions:
                x, attn = layer(x, mask, return_attn=True)
                attentions.append(attn)
            else:
                x = layer(x, mask)
        norm_out = self.norm(x)
        if return_all_attentions:
            return norm_out, attentions
        return norm_out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, pre_ln: bool = False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.sublayer_connections = nn.ModuleList([
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln),
            SublayerConnection(d_model, dropout, pre_ln=pre_ln)
        ])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        x = self.sublayer_connections[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer_connections[1](x, lambda _x: self.cross_attn(_x, memory, memory, src_mask))
        x = self.sublayer_connections[2](x, self.feed_forward)
        return x, None, None


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, d_ff: int = 2048, max_len: int = 5000, dropout: float = 0.1, pre_ln: bool = False, embedding: Optional[TransformerEmbedding] = None):
        super().__init__()
        self.embedding = embedding if embedding is not None else TransformerEmbedding(vocab_size, d_model, max_len=max_len, dropout=dropout)
        self.layers = clones(DecoderLayer(d_model, n_heads, d_ff, dropout, pre_ln=pre_ln), n_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, return_all_attentions: bool = False):
        x = self.embedding(tgt)
        for layer in self.layers:
            x, _, _ = layer(x, memory, src_mask, tgt_mask, return_attn=return_all_attentions)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


def make_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, 1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    subsequent_mask = make_subsequent_mask(tgt_len).to(tgt.device)
    return tgt_pad_mask & subsequent_mask


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 384, n_heads: int = 6, num_encoder_layers: int = 6, num_decoder_layers: int = 6, d_ff: int = 1536, max_len: int = 512, dropout: float = 0.1, pad_idx: int = 0, share_embeddings: bool = True, pre_ln: bool = False):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.share_embeddings = share_embeddings

        if share_embeddings and src_vocab_size == tgt_vocab_size:
            shared_lut = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
            enc_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx, shared_embedding=shared_lut)
            dec_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx, shared_embedding=shared_lut)
        else:
            shared_lut = None
            enc_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx)
            dec_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len=max_len, dropout=dropout, padding_idx=pad_idx)

        self.encoder = TransformerEncoder(vocab_size=src_vocab_size, d_model=d_model, n_layers=num_encoder_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout, pre_ln=pre_ln, embedding=enc_embedding)
        self.decoder = TransformerDecoder(vocab_size=tgt_vocab_size, d_model=d_model, n_layers=num_decoder_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout, pre_ln=pre_ln, embedding=dec_embedding)
        self.generator = Generator(d_model, tgt_vocab_size)

        if share_embeddings and shared_lut is not None:
            self.generator.proj.weight = shared_lut.weight

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "proj.weight" not in name:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        return self.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None):
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_idx)
        tgt_mask = make_tgt_mask(tgt, self.pad_idx)
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(dec_out)


# ==============================================================================
# 2. DATASET & TOKENIZATION
# ==============================================================================

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<2en>", "<2hi>", "<2ta>", "<2te>"]
LANG_MAP = {"en": "<2en>", "hi": "<2hi>", "ta": "<2ta>", "te": "<2te>"}

class NMTDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], tokenizer, max_len: int = 64):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        src_ctrl = LANG_MAP.get(item["tgt_lang"], "<2en>")
        src_text = f"{src_ctrl} {item['src_text']}"
        tgt_text = item["tgt_text"]

        src_ids = self.tokenizer.encode(src_text).ids[: self.max_len - 2]
        src_tokens = [self.bos_id] + src_ids + [self.eos_id]

        tgt_ids = self.tokenizer.encode(tgt_text).ids[: self.max_len - 2]
        tgt_tokens = [self.bos_id] + tgt_ids + [self.eos_id]

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)


def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    pad_id = 0
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded


# ==============================================================================
# 3. ROBUST DATASET LOCATOR & TRAINING ROUTINE
# ==============================================================================

def find_dataset_and_tokenizer() -> Tuple[Optional[Path], Optional[Path]]:
    """Recursively search for dataset files in Kaggle input or local dirs."""
    search_roots = [
        Path("/kaggle/input"),
        Path("ml_lab/kaggle/dataset"),
        Path("ml_lab/datasets"),
        Path("."),
    ]
    data_file = None
    tok_file = None

    for root in search_roots:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file():
                    if "train_multilingual.jsonl" in p.name:
                        data_file = p
                    elif "multilingual_tokenizer.json" in p.name:
                        tok_file = p

    return data_file, tok_file


def main():
    print("=" * 70)
    print("TRANSLATEME: PRODUCTION MULTILINGUAL TRANSFORMER TRAINING")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs Detected: {gpu_count} x {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    else:
        print("Notice: CUDA not detected, running on CPU.")
    print("=" * 70)

    # Locate dataset and tokenizer
    data_file, tok_file = find_dataset_and_tokenizer()
    if not data_file or not tok_file:
        print("ERROR: Dataset or Tokenizer could not be found!")
        print("Files present in search roots:")
        for r in [Path("/kaggle/input"), Path(".")]:
            if r.exists():
                for f in list(r.rglob("*"))[:30]:
                    print("  ", f)
        sys.exit(1)

    print(f"Found Dataset: {data_file} ({data_file.stat().st_size / 1e6:.2f} MB)")
    print(f"Found Tokenizer: {tok_file} ({tok_file.stat().st_size / 1e3:.1f} KB)")

    # Load Tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_file))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded Tokenizer. Vocab size: {vocab_size}")

    # Read samples
    samples = []
    opener = gzip.open if str(data_file).endswith(".gz") else open
    with opener(str(data_file), "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples):,} parallel sentences.")
    random.seed(42)
    random.shuffle(samples)

    split = int(len(samples) * 0.96)
    train_samples = samples[:split]
    val_samples = samples[split:]
    print(f"Train samples: {len(train_samples):,} | Val samples: {len(val_samples):,}")

    # Hyperparameters
    batch_size = 128 if torch.cuda.is_available() else 32
    d_model = 384
    n_heads = 6
    n_enc_layers = 6
    n_dec_layers = 6
    d_ff = 1536
    dropout = 0.1
    epochs = 10

    train_dataset = NMTDataset(train_samples, tokenizer, max_len=64)
    val_dataset = NMTDataset(val_samples, tokenizer, max_len=64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Initialize Model from Random Scratch Initializer
    model = Seq2SeqTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=n_enc_layers,
        num_decoder_layers=n_dec_layers,
        d_ff=d_ff,
        pad_idx=0,
        dropout=dropout,
        share_embeddings=True,
        pre_ln=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Seq2SeqTransformer initialized from scratch: {total_params:,} trainable parameters.")

    # Optimization
    criterion = nn.NLLLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    total_steps = len(train_loader) * epochs
    warmup_steps = min(2000, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("ml_lab/kaggle/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "custom_multilingual_transformer.pt"
    tokenizer.save(str(out_dir / "multilingual_tokenizer.json"))

    print(f"\nStarting {epochs}-epoch training...")
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        step_count = 0
        ep_start = time.time()

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                log_probs = model(src, tgt_in)
                loss = criterion(log_probs.reshape(-1, vocab_size), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            step_count += 1

            if (batch_idx + 1) % 250 == 0 or (batch_idx + 1) == len(train_loader):
                cur_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch:2d}/{epochs} | Batch {batch_idx+1:4d}/{len(train_loader)} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e}")

        avg_train_loss = total_train_loss / max(1, step_count)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]

                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    log_probs = model(src, tgt_in)
                    loss = criterion(log_probs.reshape(-1, vocab_size), tgt_out.reshape(-1))

                total_val_loss += loss.item()
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        ep_time = time.time() - ep_start
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {ep_time:.1f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": vocab_size,
                    "src_vocab_size": vocab_size,
                    "tgt_vocab_size": vocab_size,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "num_layers": n_enc_layers,
                    "num_encoder_layers": n_enc_layers,
                    "num_decoder_layers": n_dec_layers,
                    "d_ff": d_ff,
                    "dropout": dropout,
                    "pad_idx": 0,
                    "share_embeddings": True,
                    "pre_ln": False
                },
                "val_loss": best_val_loss,
                "epoch": epoch,
                "languages": ["en", "hi", "ta", "te"],
                "control_tokens": SPECIAL_TOKENS,
                "weights_source": "Trained 100% from scratch on Kaggle Cloud GPUs"
            }, str(best_ckpt_path))
            print(f"  [*] Best checkpoint saved! (Val Loss: {best_val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes! Best model saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
