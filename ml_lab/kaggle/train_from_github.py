#!/usr/bin/env python3
"""
TranslateMe Option 1 (113.29M 8E/4D Transformer) — GitHub-Driven Kaggle 2x T4 Trainer
====================================================================================
Designed specifically for the GitHub-Pull + Kaggle Dual-GPU ("Persistence: Files only") workflow:
- Imports directly from repository modules (zero duplicated model code).
- True Multi-GPU DistributedDataParallel (DDP) via torch.multiprocessing.spawn across 2x Tesla T4
  (automatically falls back to single GPU / MPS / CPU when world_size == 1).
- Option 1 SOTA Config: 8 Encoder Layers, 4 Decoder Layers, d_model=768, 12 heads, d_ff=2048,
  24,576 ByteLevel BPE vocab -> 113,292,800 (113.29M) trainable parameters.
- Full 12-Direction Any-to-Any Data (EN <-> HI <-> TA <-> TE) with Pre-Tokenized Cache + Length Bucketing.
- Bulletproof Checkpointing & Kaggle Persistence:
  * Auto-saves full resumable state every 1,000 optimizer steps (--save_every 1000)
  * Instant graceful pause & atomic save on Jupyter Stop (SIGINT/SIGTERM) or pause.flag
  * Wall-clock safety auto-pause at 11h 20m (--max_run_minutes 680) before Kaggle's 12h kill
  * Stores persistent state under /kaggle/working/translateme_state (outside GitHub clone)
  * Stores ephemeral parquet downloads in /tmp/translateme_cache to protect Kaggle's 19.5 GB disk quota
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time
import json
import math
import signal
import shutil
import random
import argparse
import contextlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.amp import autocast, GradScaler

try:
    import numpy as np
except ImportError:
    np = None

from ml_lab.custom_transformer.modern_transformer import ModernSeq2SeqTransformer, ModernTransformerConfig
from ml_lab.tokenizer.multilingual_tokenizer import MultilingualSubwordTokenizer
from ml_lab.datasets.build_100m_dataset import build_optimal_dataset
from ml_lab.training.resumable_trainer import (
    CosineWarmupScheduler,
    configure_adamw_optimizer,
    get_preferred_amp_dtype,
    format_duration,
)


# ==============================================================================
# 1. PRE-TOKENIZED COMPACT DATASET & DISTRIBUTED LENGTH-BUCKETED SAMPLER
# ==============================================================================

class PreTokenizedNMTDataset(Dataset):
    """
    Loads 12-direction parallel JSONL once, batch-tokenizes via Rust HuggingFace tokenizer,
    caches compact int16/int32 flat tensors into persistent storage, and serves batches with ~0 CPU overhead.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: MultilingualSubwordTokenizer,
        max_len: int = 128,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        self.bos_id = getattr(tokenizer, "bos_id", tokenizer.sos_id)
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
        self.max_len = max_len
        self.tok_dtype = torch.int16 if tokenizer.vocab_size <= 32767 else torch.int32

        if cache_path and os.path.exists(cache_path) and max_samples is None:
            print(f"  [Cache Hit] Loading pre-tokenized tensors: {Path(cache_path).name}")
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.src_data = cached["src_data"]
            self.src_offsets = cached["src_offsets"]
            self.tgt_data = cached["tgt_data"]
            self.tgt_offsets = cached["tgt_offsets"]
            self.lengths = cached["lengths"]
            return

        t0 = time.time()
        print(f"  [Pre-Tokenize] Batch-tokenizing {Path(jsonl_path).name}...")
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
            src_enc = [e.ids for e in tokenizer.tokenizer.encode_batch(sources[start:end])]
            tgt_enc = [e.ids for e in tokenizer.tokenizer.encode_batch(targets[start:end])]

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

        print(f"  [Pre-Tokenize] Encoded {n:,} pairs in {time.time() - t0:.1f}s.")
        if cache_path and max_samples is None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = f"{cache_path}.tmp"
            torch.save({
                "src_data": self.src_data,
                "src_offsets": self.src_offsets,
                "tgt_data": self.tgt_data,
                "tgt_offsets": self.tgt_offsets,
                "lengths": self.lengths,
            }, tmp_cache)
            os.replace(tmp_cache, cache_path)
            print(f"  [Cache Saved] Compact cache written: {cache_path} ({Path(cache_path).stat().st_size / (1024*1024):.1f} MB)")

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s0, s1 = int(self.src_offsets[idx]), int(self.src_offsets[idx + 1])
        t0, t1 = int(self.tgt_offsets[idx]), int(self.tgt_offsets[idx + 1])
        return self.src_data[s0:s1].long(), self.tgt_data[t0:t1].long()


class DistributedBucketBatchSampler(Sampler[List[int]]):
    """
    Deterministic Length-Bucketed Sampler (Fixed batch_size or Token-Budget Dynamic Batching)
    compatible with both single-GPU and multi-GPU DDP.
    When `max_tokens > 0`, short sentences scale up to `max_batch_size` (saturating T4 Tensor Cores)
    while long sentences automatically shrink batch size to prevent CUDA OOM.
    """
    def __init__(
        self,
        lengths: List[int],
        batch_size: int = 32,
        max_tokens: int = 0,
        max_batch_size: int = 128,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        mega_batch_size: int = 4096
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.rank = rank
        self.world_size = max(1, world_size)
        self.seed = seed
        self.mega_batch_size = mega_batch_size
        self.epoch = 0
        self._cached_epoch: Optional[int] = None
        self._cached_batches: Optional[List[List[int]]] = None

    def set_epoch(self, epoch: int):
        if self._cached_epoch != epoch:
            self._cached_batches = None
        self.epoch = epoch

    def _build_global_batches(self) -> List[List[int]]:
        if self._cached_batches is not None and self._cached_epoch == self.epoch:
            return self._cached_batches

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=g).tolist()

        batches: List[List[int]] = []
        for i in range(0, len(indices), self.mega_batch_size):
            chunk = indices[i:i + self.mega_batch_size]
            chunk.sort(key=lambda idx: self.lengths[idx])

            if self.max_tokens > 0:
                cur_batch: List[int] = []
                cur_max_len = 0
                for idx in chunk:
                    l = self.lengths[idx]
                    next_max = max(cur_max_len, l)
                    if cur_batch and ((len(cur_batch) + 1) * next_max > self.max_tokens or len(cur_batch) >= self.max_batch_size):
                        batches.append(cur_batch)
                        cur_batch = [idx]
                        cur_max_len = l
                    else:
                        cur_batch.append(idx)
                        cur_max_len = next_max
                if cur_batch:
                    batches.append(cur_batch)
            else:
                for j in range(0, len(chunk), self.batch_size):
                    batches.append(chunk[j:j + self.batch_size])

        usable = (len(batches) // self.world_size) * self.world_size
        if usable == 0 and batches:
            while len(batches) < self.world_size:
                batches.append(batches[-1])
            usable = self.world_size

        perm = torch.randperm(len(batches), generator=g).tolist()[:usable]
        self._cached_batches = [batches[idx] for idx in perm]
        self._cached_epoch = self.epoch
        return self._cached_batches

    def __len__(self) -> int:
        return len(self._build_global_batches()) // self.world_size

    def __iter__(self) -> Iterator[List[int]]:
        global_batches = self._build_global_batches()
        for step_idx in range(self.rank, len(global_batches), self.world_size):
            yield global_batches[step_idx]


class TensorCollateFn:
    """Top-level picklable collate functor compatible with Python 3.13+ spawn DataLoader workers."""
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        src_list = [b[0] for b in batch]
        tgt_in_list = [b[1][:-1] for b in batch]
        tgt_out_list = [b[1][1:] for b in batch]
        return {
            "src": nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=self.pad_id),
            "tgt_in": nn.utils.rnn.pad_sequence(tgt_in_list, batch_first=True, padding_value=self.pad_id),
            "tgt_out": nn.utils.rnn.pad_sequence(tgt_out_list, batch_first=True, padding_value=self.pad_id),
        }


def build_tensor_collate_fn(pad_id: int = 0) -> TensorCollateFn:
    return TensorCollateFn(pad_id=pad_id)


# ==============================================================================
# 2. DATA & TOKENIZER PREPARATION (RUNS ONCE ON RANK 0 BEFORE SPAWNING DDP)
# ==============================================================================

def prepare_artifacts(args) -> Tuple[Path, Path, str, str]:
    """Prepares dataset, tokenizer (24,576 vocab), and pre-tokenized tensor caches."""
    work_dir = Path(args.work_dir)
    ckpt_dir = work_dir / "checkpoints"
    data_dir = work_dir / "data"
    tok_dir = work_dir / "tokenizer"
    cache_dir = Path(args.cache_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_tok_cache = data_dir / "train_tok_v2.pt"
    val_tok_cache = data_dir / "val_tok_v2.pt"
    tokenizer_path = str(tok_dir / "multilingual_tokenizer.json")

    raw_jsonl_dir = cache_dir / "jsonl"
    train_file = raw_jsonl_dir / "train.jsonl"
    val_file = raw_jsonl_dir / "val.jsonl"

    # Build raw 12-direction JSONL only if pre-tokenized cache & tokenizer don't both exist yet
    if not (train_tok_cache.exists() and val_tok_cache.exists() and os.path.exists(tokenizer_path)):
        if not train_file.exists() or not val_file.exists():
            print("\n[Phase 1] Building 12-direction Any-to-Any Dataset (EN <-> HI <-> TA <-> TE)...")
            if args.dataset_size == "optimal":
                build_optimal_dataset(
                    target_hi=1_000_000, target_ta=700_000, target_te=700_000, target_indic_pair=400_000,
                    output_dir=str(raw_jsonl_dir), cache_dir=str(cache_dir / "parquets")
                )
            elif args.dataset_size == "medium":
                build_optimal_dataset(
                    target_hi=150_000, target_ta=100_000, target_te=100_000, target_indic_pair=60_000,
                    output_dir=str(raw_jsonl_dir), cache_dir=str(cache_dir / "parquets")
                )
            else:
                build_optimal_dataset(
                    target_hi=5_000, target_ta=2_500, target_te=2_500, target_indic_pair=1_500,
                    val_cap_per_dir=300, test_cap_per_dir=300,
                    output_dir=str(raw_jsonl_dir), cache_dir=str(cache_dir / "parquets")
                )

        if not os.path.exists(tokenizer_path):
            print(f"\n[Phase 2] Training {args.vocab_size:,}-Vocab Multilingual BPE Tokenizer on train split only...")
            tokenizer = MultilingualSubwordTokenizer()
            texts = []
            with open(train_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if idx >= 400_000:
                        break
                    d = json.loads(line)
                    texts.append(d["source"])
                    texts.append(d["target"])
            tokenizer.train_from_iterator(texts, vocab_size=args.vocab_size, min_frequency=2)
            tokenizer.save(tokenizer_path)
            print(f"  Tokenizer saved -> {tokenizer_path} (Vocab: {tokenizer.vocab_size:,})")
        else:
            tokenizer = MultilingualSubwordTokenizer.load(tokenizer_path)

        print("\n[Phase 3] Creating persistent compact pre-tokenized caches in /kaggle/working...")
        PreTokenizedNMTDataset(str(train_file), tokenizer, max_len=args.max_len, cache_path=str(train_tok_cache))
        PreTokenizedNMTDataset(str(val_file), tokenizer, max_len=args.max_len, cache_path=str(val_tok_cache))
    else:
        print(f"\n[Fast Resume] Using persistent tokenizer & pre-tokenized caches from {work_dir}")

    return train_tok_cache, val_tok_cache, tokenizer_path, str(ckpt_dir)


# ==============================================================================
# 3. DDP WORKER TRAINING LOOP
# ==============================================================================

class PauseMonitor:
    """Thread/signal-safe interrupt monitor so all DDP ranks pause together without NCCL deadlock."""
    def __init__(self, max_run_minutes: float, pause_paths: List[Path]):
        self.start_time = time.time()
        self.max_seconds = max_run_minutes * 60.0 if max_run_minutes > 0 else float("inf")
        self.pause_paths = pause_paths
        self.signal_received = False

        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.signal_received = True

    def should_pause(self, rank: int, world_size: int, device: torch.device) -> Tuple[bool, str]:
        reason = ""
        local_flag = 0
        if rank == 0:
            if self.signal_received:
                local_flag = 1
                reason = "SIGINT/SIGTERM (Stop button)"
            elif (time.time() - self.start_time) >= self.max_seconds:
                local_flag = 2
                reason = f"Safety wall-clock limit ({self.max_seconds / 3600:.2f}h reached before 12h timeout)"
            else:
                for p in self.pause_paths:
                    if p.exists():
                        local_flag = 3
                        reason = f"pause.flag detected ({p})"
                        try:
                            p.unlink()
                        except Exception:
                            pass
                        break

        if world_size > 1 and dist.is_initialized():
            t = torch.tensor([local_flag], dtype=torch.int32, device=device)
            dist.broadcast(t, src=0)
            local_flag = int(t.item())
        return local_flag > 0, reason


def save_checkpoint_rank0(
    ckpt_dir: Path,
    filename: str,
    raw_model: ModernSeq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    tokenizer_path: str,
    epoch: int,
    epoch_batch_idx: int,
    global_step: int,
    best_val_loss: float,
    train_history: List[float],
    val_history: List[float],
    is_best: bool = False,
    mirror_kaggle_root: bool = True
):
    """Writes full resumable state atomically, plus lightweight inference-ready checkpoint_best.pt."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / filename
    tmp_path = ckpt_dir / f"{filename}.tmp"

    cfg_dict = dict(raw_model.config.__dict__)
    cfg_dict["model_type"] = "modern"

    state = {
        "epoch": epoch,
        "epoch_batch_idx": epoch_batch_idx,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "config": cfg_dict,
        "tokenizer_path": tokenizer_path,
        "train_loss_history": train_history,
        "val_loss_history": val_history,
        "rng_state": {
            "cpu": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state() if np is not None else None,
        },
        "timestamp": time.time(),
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"[CHECKPOINT] Atomic save: {save_path} (Epoch {epoch+1}, Batch {epoch_batch_idx}, Step {global_step})")

    if is_best:
        best_path = ckpt_dir / "checkpoint_best.pt"
        tmp_best = ckpt_dir / "checkpoint_best.pt.tmp"
        inference_state = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "model_state_dict": raw_model.state_dict(),
            "config": cfg_dict,
            "tokenizer_path": tokenizer_path,
            "train_loss_history": train_history,
            "val_loss_history": val_history,
            "timestamp": time.time(),
        }
        torch.save(inference_state, tmp_best)
        os.replace(tmp_best, best_path)
        print(f"[*] Updated inference best model: {best_path} (Val Loss: {best_val_loss:.4f})")

        if mirror_kaggle_root and os.path.exists("/kaggle/working"):
            try:
                shutil.copy2(best_path, "/kaggle/working/checkpoint_best.pt")
                if os.path.exists(tokenizer_path):
                    shutil.copy2(tokenizer_path, "/kaggle/working/multilingual_tokenizer.json")
            except Exception:
                pass

    # Keep only the latest 2 epoch snapshots to stay comfortably within Kaggle's 19.5 GB limit
    epoch_files = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    for old in epoch_files[:-2]:
        try:
            old.unlink()
        except Exception:
            pass


@torch.no_grad()
def evaluate_distributed(
    model_engine: nn.Module,
    raw_model: ModernSeq2SeqTransformer,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    world_size: int,
    max_batches: Optional[int] = None
) -> float:
    model_engine.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=raw_model.pad_idx, reduction="sum")
    loss_sum = 0.0
    tok_sum = 0

    for idx, batch in enumerate(val_loader):
        if max_batches is not None and idx >= max_batches:
            break
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model_engine(src, tgt_in)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
        loss_sum += loss.item()
        tok_sum += int((tgt_out != raw_model.pad_idx).sum().item())

    if world_size > 1 and dist.is_initialized():
        stats = torch.tensor([loss_sum, float(tok_sum)], dtype=torch.float64, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, tok_sum = stats[0].item(), int(stats[1].item())

    model_engine.train()
    return loss_sum / max(tok_sum, 1)


@torch.no_grad()
def run_12dir_probes(raw_model: ModernSeq2SeqTransformer, tokenizer: MultilingualSubwordTokenizer):
    probes = [
        ("Hello, how are you today?", "hi"),
        ("नमस्ते, आज आप कैसे हैं?", "ta"),
        ("வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?", "te"),
        ("నమస్కారం, ఈరోజు మీరు ఎలా ఉన్నారు?", "en"),
    ]
    print("\n[LIVE ANY-TO-ANY PROBES]")
    for text, tgt_lang in probes:
        src_ids = tokenizer.encode(text, add_special_tokens=True)
        src_t = torch.tensor([src_ids], dtype=torch.long)
        ctrl_id = tokenizer.get_control_token_id(tgt_lang)
        gen_t = raw_model.greedy_decode(src_t, sos_idx=getattr(tokenizer, "bos_id", tokenizer.sos_id), eos_idx=tokenizer.eos_id, max_len=48, prefix_tokens=[ctrl_id])
        out_str = tokenizer.decode(gen_t[0].tolist(), skip_special_tokens=True)
        print(f"  [->{tgt_lang}] '{text}' => '{out_str}'")
    print()


def ddp_worker(rank: int, world_size: int, args, train_tok_cache: str, val_tok_cache: str, tokenizer_path: str, ckpt_dir_str: str):
    use_ddp = world_size > 1
    if use_ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.ddp_port))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.epochs <= 0:
        raise ValueError("--epochs must be >= 1")

    # Auto-fit token budget / batch_size on <=16GB GPUs (e.g. Kaggle Tesla T4 14.6 GiB)
    if device.type == "cuda" and not args.no_auto_vram_fit:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        if vram_gb < 17.0:
            if args.max_tokens > 3600:
                if rank == 0:
                    print(f"[Auto-VRAM Fit] {vram_gb:.1f} GB GPU detected: clamping max_tokens {args.max_tokens}->3600")
                args.max_tokens = 3600
            elif args.max_tokens == 0 and args.batch_size > 32:
                factor = math.ceil(args.batch_size / 32)
                new_bs = max(16, args.batch_size // factor)
                new_accum = args.grad_accum * factor
                if rank == 0:
                    print(f"[Auto-VRAM Fit] {vram_gb:.1f} GB GPU detected: adjusting batch_per_gpu {args.batch_size}->{new_bs}, grad_accum {args.grad_accum}->{new_accum}")
                args.batch_size = new_bs
                args.grad_accum = new_accum

    ckpt_dir = Path(ckpt_dir_str)
    tokenizer = MultilingualSubwordTokenizer.load(tokenizer_path)

    # Load pre-tokenized compact caches into memory
    train_ds = PreTokenizedNMTDataset("", tokenizer, max_len=args.max_len, cache_path=train_tok_cache)
    val_ds = PreTokenizedNMTDataset("", tokenizer, max_len=args.max_len, cache_path=val_tok_cache)

    collate_fn = build_tensor_collate_fn(pad_id=tokenizer.pad_id)
    train_sampler = DistributedBucketBatchSampler(
        train_ds.lengths,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        rank=rank,
        world_size=world_size,
        seed=args.seed
    )
    val_sampler = DistributedBucketBatchSampler(
        val_ds.lengths,
        batch_size=min(args.batch_size, 32),
        max_tokens=0,
        rank=rank,
        world_size=world_size,
        seed=args.seed
    )
    # Pre-tokenized int16 tensors slice directly in RAM in microseconds; num_workers=0 avoids
    # duplicating the 837 MB tensor cache across spawn DataLoader workers in DDP.
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler, collate_fn=collate_fn,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Option 1 Model: 8 Enc / 4 Dec, d_model=768, 12 heads, d_ff=2048, vocab=24,576 (~113.29M params)
    config = ModernTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_idx=tokenizer.pad_id,
        bos_idx=getattr(tokenizer, "bos_id", tokenizer.sos_id),
        eos_idx=tokenizer.eos_id,
        use_qk_norm=True,
        use_rope=True,
        share_embeddings=True,
        use_layerscale=True,
        soft_cap=30.0,
    )
    raw_model = ModernSeq2SeqTransformer(config).to(device)
    if rank == 0:
        p_cnt = raw_model.count_parameters()
        batch_desc = f"max_tokens={args.max_tokens}/GPU (cap {args.max_batch_size})" if args.max_tokens > 0 else f"batch_per_gpu={args.batch_size}"
        print(f"\n[Model] Option 1 Asymmetric ModernSeq2SeqTransformer ({args.enc_layers}E/{args.dec_layers}D, d={args.d_model}, vocab={tokenizer.vocab_size:,}): {p_cnt:,} ({p_cnt/1e6:.2f}M params)")
        print(f"[Hardware] world_size={world_size} | {batch_desc} | grad_accum={args.grad_accum} | DDP no_sync=True")

    model_engine: nn.Module = DDP(raw_model, device_ids=[rank], output_device=rank) if use_ddp else raw_model

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    optimizer = configure_adamw_optimizer(raw_model, lr=args.lr, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=args.warmup, total_steps=total_steps, peak_lr=args.lr)

    amp_dtype, needs_scaler = get_preferred_amp_dtype(device)
    use_amp = (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp and needs_scaler)
    sum_criterion = nn.CrossEntropyLoss(ignore_index=raw_model.pad_idx, label_smoothing=0.1, reduction="sum")

    start_epoch = 0
    resume_batch_idx = 0
    global_step = 0
    best_val_loss = float("inf")
    train_history: List[float] = []
    val_history: List[float] = []

    # Auto-Resume detection
    resume_candidates = []
    if args.resume and args.resume != "auto":
        resume_candidates.append(args.resume)
    else:
        resume_candidates.extend([
            str(ckpt_dir / "checkpoint_latest.pt"),
            "/kaggle/working/checkpoint_latest.pt",
        ])

    for cand in resume_candidates:
        if cand and os.path.exists(cand):
            chk = torch.load(cand, map_location=device, weights_only=False)
            raw_model.load_state_dict(chk["model_state_dict"])
            if chk.get("optimizer_state_dict"):
                optimizer.load_state_dict(chk["optimizer_state_dict"])
            if chk.get("scheduler_state_dict"):
                scheduler.load_state_dict(chk["scheduler_state_dict"])
            if scaler.is_enabled() and chk.get("scaler_state_dict"):
                scaler.load_state_dict(chk["scaler_state_dict"])
            start_epoch = chk.get("epoch", 0)
            resume_batch_idx = chk.get("epoch_batch_idx", 0)
            global_step = chk.get("global_step", 0)
            best_val_loss = chk.get("best_val_loss", float("inf"))
            train_history = chk.get("train_loss_history", [])
            val_history = chk.get("val_loss_history", [])
            if rank == 0:
                print(f"[AUTO-RESUME] Loaded {cand} -> Epoch {start_epoch+1}, Batch {resume_batch_idx}, Step {global_step} (Best Val: {best_val_loss:.4f})")
            break

    pause_monitor = PauseMonitor(
        max_run_minutes=args.max_run_minutes,
        pause_paths=[Path("pause.flag"), Path("/kaggle/working/pause.flag"), ckpt_dir / "pause.flag"],
    )

    epoch_batch_idx = resume_batch_idx
    halted = False

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        optimizer.zero_grad(set_to_none=True)

        num_batches = len(train_loader)
        skip_until = epoch_batch_idx if epoch == start_epoch else 0
        if rank == 0 and skip_until > 0:
            print(f"  [Mid-Epoch Resume] Skipping already completed batches 0..{skip_until-1} in Epoch {epoch+1}...")

        run_start = time.time()
        running_loss = 0.0
        batches_in_window = 0
        batches_processed_run = 0
        tokens_processed_run = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_until:
                continue

            # Check pause cleanly at gradient-accumulation boundaries so zero micro-batch gradients are lost
            if batch_idx % args.grad_accum == 0:
                should_stop, stop_reason = pause_monitor.should_pause(rank, world_size, device)
                if should_stop:
                    epoch_batch_idx = batch_idx
                    if rank == 0:
                        print(f"\n[PAUSE TRIGGERED] {stop_reason}. Saving atomic checkpoint before exit...")
                        save_checkpoint_rank0(
                            ckpt_dir, "checkpoint_latest.pt", raw_model, optimizer, scheduler, scaler,
                            tokenizer_path, epoch, epoch_batch_idx, global_step, best_val_loss,
                            train_history, val_history, is_best=False
                        )
                        print("[PAUSED CLEANLY] You can reset your Kaggle 12h session now; next run resumes at this exact batch.")
                    halted = True
                    break

            src = batch["src"].to(device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(device, non_blocking=True)
            tgt_out = batch["tgt_out"].to(device, non_blocking=True)

            valid_mask = tgt_out != raw_model.pad_idx
            batch_tokens = int(valid_mask.sum().item())
            tokens_processed_run += batch_tokens * world_size

            chunk_start = (batch_idx // args.grad_accum) * args.grad_accum
            actual_chunk_size = min(args.grad_accum, num_batches - chunk_start)
            is_step_boundary = ((batch_idx + 1) % args.grad_accum == 0) or ((batch_idx + 1) == num_batches)
            sync_ctx = contextlib.nullcontext() if (not use_ddp or is_step_boundary) else model_engine.no_sync()

            with sync_ctx:
                with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model_engine(src, tgt_in)
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = tgt_out.view(-1)
                    valid_flat = valid_mask.view(-1)
                    # Compute loss strictly over non-padding tokens (saves ~25-40% softmax/CE VRAM)
                    # Soft-cap=30.0 already bounds logits cleanly without extra FP32 Z-loss copy
                    if valid_flat.any():
                        batch_sum_loss = sum_criterion(flat_logits[valid_flat], flat_targets[valid_flat])
                    else:
                        batch_sum_loss = flat_logits.sum() * 0.0
                    raw_loss = batch_sum_loss / max(batch_tokens, 1)
                    loss = raw_loss / actual_chunk_size

                scaler.scale(loss).backward()

            running_loss += raw_loss.item()
            batches_in_window += 1
            batches_processed_run += 1
            epoch_batch_idx = batch_idx + 1

            if is_step_boundary:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_engine.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    val_loss = evaluate_distributed(
                        model_engine, raw_model, val_loader, device, amp_dtype, use_amp, world_size, max_batches=40
                    )
                    val_history.append(val_loss)
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    if rank == 0:
                        print(f"  [Mid-Eval @ Step {global_step}] Val Loss: {val_loss:.4f} | PPL: {math.exp(min(val_loss, 20)):.2f} | Best: {best_val_loss:.4f}")
                        save_checkpoint_rank0(
                            ckpt_dir, "checkpoint_latest.pt", raw_model, optimizer, scheduler, scaler,
                            tokenizer_path, epoch, epoch_batch_idx, global_step, best_val_loss,
                            train_history, val_history, is_best=is_best
                        )
                elif global_step % args.save_every == 0 and rank == 0:
                    save_checkpoint_rank0(
                        ckpt_dir, "checkpoint_latest.pt", raw_model, optimizer, scheduler, scaler,
                        tokenizer_path, epoch, epoch_batch_idx, global_step, best_val_loss,
                        train_history, val_history, is_best=False
                    )

            if rank == 0 and ((batch_idx + 1) % args.log_every == 0 or (batch_idx + 1) == num_batches):
                elapsed = time.time() - run_start
                batches_done = batch_idx + 1
                bps = batches_processed_run / max(elapsed, 0.001)
                ep_left = (num_batches - batches_done) / max(bps, 0.001)
                tot_left = (max(args.epochs - (epoch + 1), 0) * num_batches + (num_batches - batches_done)) / max(bps, 0.001)
                tok_sec = tokens_processed_run / max(elapsed, 0.001)
                avg_loss = running_loss / max(batches_in_window, 1)
                ppl = math.exp(min(avg_loss, 20.0))
                lr = optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch+1:2d}/{args.epochs:2d} | "
                    f"Batch {batches_done:5d}/{num_batches:5d} ({batches_done/num_batches*100:4.1f}%) | "
                    f"Step {global_step:6d} | Loss: {avg_loss:5.3f} | PPL: {ppl:6.2f} | "
                    f"LR: {lr:.2e} | Speed: {tok_sec:6.0f} tok/s | "
                    f"Epoch Left: {format_duration(ep_left)} | Total ETA: {format_duration(tot_left)}"
                )
                train_history.append(avg_loss)
                running_loss = 0.0
                batches_in_window = 0

                try:
                    prog = ckpt_dir / "training_progress.json"
                    prog_tmp = ckpt_dir / "training_progress.json.tmp"
                    with open(prog_tmp, "w", encoding="utf-8") as f:
                        json.dump({
                            "epoch": epoch + 1, "total_epochs": args.epochs,
                            "batch": batches_done, "total_batches": num_batches,
                            "global_step": global_step, "loss": round(avg_loss, 4),
                            "perplexity": round(ppl, 2), "tok_per_sec": int(tok_sec),
                            "epoch_left": format_duration(ep_left), "total_eta": format_duration(tot_left),
                            "best_val_loss": round(best_val_loss, 4) if best_val_loss != float("inf") else None,
                            "timestamp": time.time(),
                        }, f, indent=2)
                    os.replace(prog_tmp, prog)
                except Exception:
                    pass

        if halted:
            break

        # End of epoch full validation
        val_loss = evaluate_distributed(model_engine, raw_model, val_loader, device, amp_dtype, use_amp, world_size)
        val_history.append(val_loss)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        epoch_batch_idx = 0
        if rank == 0:
            print(f"\n=== Epoch {epoch+1}/{args.epochs} Complete | Val Loss: {val_loss:.4f} | Val PPL: {math.exp(min(val_loss, 20)):.2f} | Best: {best_val_loss:.4f} ===")
            save_checkpoint_rank0(
                ckpt_dir, "checkpoint_latest.pt", raw_model, optimizer, scheduler, scaler,
                tokenizer_path, epoch + 1, 0, global_step, best_val_loss,
                train_history, val_history, is_best=is_best
            )
            save_checkpoint_rank0(
                ckpt_dir, f"checkpoint_epoch_{epoch+1}.pt", raw_model, optimizer, scheduler, scaler,
                tokenizer_path, epoch + 1, 0, global_step, best_val_loss,
                train_history, val_history, is_best=False
            )
            run_12dir_probes(raw_model, tokenizer)

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def parse_args(args_list: Optional[List[str]] = None):
    on_kaggle = os.path.exists("/kaggle/working")
    default_work = "/kaggle/working/translateme_state" if on_kaggle else "ml_lab/translateme_state"
    default_cache = "/tmp/translateme_cache" if on_kaggle else "ml_lab/datasets/real_corpus"

    parser = argparse.ArgumentParser(description="Option 1 113.29M Multi-GPU DDP Trainer (GitHub -> Kaggle)")
    parser.add_argument("--work_dir", type=str, default=default_work, help="Persistent state directory (checkpoints, tok, cache)")
    parser.add_argument("--cache_dir", type=str, default=default_cache, help="Ephemeral download directory (keeps /kaggle/working clean)")
    parser.add_argument("--dataset_size", choices=["optimal", "medium", "smoke"], default="optimal")
    parser.add_argument("--vocab_size", type=int, default=24576)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--enc_layers", type=int, default=8)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=3600, help="Token-budget dynamic batching per GPU (0 disables and uses --batch_size)")
    parser.add_argument("--max_batch_size", type=int, default=128, help="Max sentences per micro-batch under token-budget batching")
    parser.add_argument("--batch_size", type=int, default=32, help="Fallback fixed batch size per GPU when --max_tokens=0")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--no_auto_vram_fit", action="store_true", help="Disable automatic <=16GB T4 batch splitting")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=1000, help="Save atomic checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=2500, help="Run mid-epoch validation every N steps")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--max_run_minutes", type=float, default=680.0, help="Auto-pause cleanly before Kaggle 12h timeout (default 11h20m)")
    parser.add_argument("--resume", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddp_port", type=int, default=29505)
    parser.add_argument("--no_ddp", action="store_true", help="Force single-process training even with multiple GPUs")

    if args_list is not None:
        args, _ = parser.parse_known_args(args_list)
    else:
        args, _ = parser.parse_known_args()
    return args


def main(args_list: Optional[List[str]] = None):
    args = parse_args(args_list)
    train_tok_cache, val_tok_cache, tokenizer_path, ckpt_dir = prepare_artifacts(args)

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    world_size = gpu_count if (gpu_count > 1 and not args.no_ddp) else 1

    print("=" * 80)
    print(f"LAUNCHING OPTION 1 TRAINING | GPUs: {gpu_count} | DDP world_size: {world_size}")
    print(f"Persistence Dir: {args.work_dir} | Save every: {args.save_every} steps | Safety Stop: {args.max_run_minutes:.0f}m")
    print("=" * 80)

    if world_size > 1:
        parent_interrupted = False
        orig_sigint = signal.getsignal(signal.SIGINT)
        orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _parent_sig_handler(signum, frame):
            nonlocal parent_interrupted
            if not parent_interrupted:
                parent_interrupted = True
                try:
                    (Path(ckpt_dir) / "pause.flag").write_text("signal_pause", encoding="utf-8")
                except Exception:
                    pass
                print("\n[PAUSE] Interrupt received — waiting for GPU workers to save atomic checkpoint cleanly...", flush=True)

        signal.signal(signal.SIGINT, _parent_sig_handler)
        signal.signal(signal.SIGTERM, _parent_sig_handler)
        try:
            ctx = mp.spawn(
                ddp_worker,
                args=(world_size, args, str(train_tok_cache), str(val_tok_cache), tokenizer_path, ckpt_dir),
                nprocs=world_size,
                join=False,
            )
            while not ctx.join(timeout=1.0):
                pass
        finally:
            signal.signal(signal.SIGINT, orig_sigint)
            signal.signal(signal.SIGTERM, orig_sigterm)
    else:
        ddp_worker(0, 1, args, str(train_tok_cache), str(val_tok_cache), tokenizer_path, ckpt_dir)


if __name__ == "__main__":
    main()
