"""
Complete Multilingual Transformer Training Pipeline
Section 5 of 'Attention Is All You Need' (Vaswani et al., 2017)
Trains randomly initialized PyTorch Seq2SeqTransformer with:
- Subword BPE tokenizer
- Multilingual target control tokens (<2en>, <2hi>, <2ta>, <2te>)
- Shifted target teacher forcing
- Padding-ignored Cross-Entropy with Label Smoothing
- AdamW optimizer + Noam learning rate scheduler
- Validation loss monitoring & best checkpoint saving
"""

import os
import math
import json
import random
import time
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml_lab.custom_transformer import (
    Seq2SeqTransformer,
    TransformerConfig,
    make_src_mask,
    make_tgt_mask
)
from ml_lab.tokenizer import MultilingualSubwordTokenizer
from ml_lab.datasets import (
    generate_multilingual_training_set,
    get_all_monolingual_sentences,
    split_parallel_dataset,
    clean_and_prepare_parallel_data
)


class NoamLRScheduler:
    """
    Noam Learning Rate Schedule from 'Attention Is All You Need' Section 5.3:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 400, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.factor * (self.d_model ** (-0.5)) * min(
            self.step_num ** (-0.5),
            self.step_num * (self.warmup_steps ** (-1.5))
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class MultilingualNMTDataset(Dataset):
    """
    Dataset of (source_sentence, target_sentence, target_lang_code).
    Tokenizes using shared BPE vocabulary and adds target control token.
    """
    def __init__(
        self,
        samples: List[Tuple[str, str, str]],
        tokenizer: MultilingualSubwordTokenizer,
        max_seq_len: int = 64
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text, tgt_lang = self.samples[idx]

        # Source tokens: <sos> ... <eos>
        src_ids = self.tokenizer.encode(src_text, add_special_tokens=True)
        # Target tokens: <sos> <2{lang}> ... <eos>
        tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=True, target_lang=tgt_lang)

        # Truncate if necessary
        if len(src_ids) > self.max_seq_len:
            src_ids = src_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]
        if len(tgt_ids) > self.max_seq_len:
            tgt_ids = tgt_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_nmt_batch(batch, pad_id: int):
    """Pads variable-length sequences dynamically in batch."""
    src_list, tgt_list = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_id)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_multilingual_model(
    epochs: int = 40,
    batch_size: int = 16,
    d_model: int = 256,
    n_heads: int = 4,
    num_layers: int = 3,
    d_ff: int = 1024,
    dropout: float = 0.1,
    warmup_steps: int = 200,
    learning_rate: float = 5e-4,
    device: Optional[str] = None,
    save_checkpoint_dir: str = "ml_lab/checkpoints",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Full training routine for the custom multilingual Transformer.
    """
    set_seed(seed)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"Training Custom Multilingual Transformer from Scratch")
    print(f"Device: {device} | Architecture: {num_layers}L-{n_heads}H-{d_model}D-{d_ff}FF")
    print(f"========================================================")

    # 1. Prepare and Train Multilingual Tokenizer
    tokenizer_path = "ml_lab/tokenizer/multilingual_tokenizer.json"
    tokenizer = MultilingualSubwordTokenizer()
    monolingual_sentences = get_all_monolingual_sentences()

    print(f"\n[Phase 1] Training Subword BPE Tokenizer on {len(monolingual_sentences)} multilingual sentences...")
    tokenizer.train_from_iterator(monolingual_sentences, vocab_size=2000, min_frequency=1)
    tokenizer.save(tokenizer_path)
    print(f"Subword Tokenizer saved to: {tokenizer_path} (Vocab size: {tokenizer.vocab_size})")

    # 2. Prepare Multilingual Parallel Dataset
    all_samples = generate_multilingual_training_set()
    print(f"\n[Phase 2] Curated {len(all_samples)} parallel pairs across en, hi, ta, te")

    # Split: 80% train, 20% validation
    random.Random(seed).shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    print(f"Dataset Split -> Train: {len(train_samples)} | Validation: {len(val_samples)}")

    train_ds = MultilingualNMTDataset(train_samples, tokenizer, max_seq_len=64)
    val_ds = MultilingualNMTDataset(val_samples, tokenizer, max_seq_len=64)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_nmt_batch(b, tokenizer.pad_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_nmt_batch(b, tokenizer.pad_id)
    )

    # 3. Initialize Model from Scratch (Random weights)
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=tokenizer.pad_id,
        share_embeddings=True
    )
    model = Seq2SeqTransformer.from_config(config).to(device)
    param_counts = model.count_parameters()
    print(f"Model initialized randomly. Trainable parameters: {param_counts['trainable_parameters']:,}")

    # 4. Optimizer, Scheduler, and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    scheduler = NoamLRScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, factor=1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, label_smoothing=0.1)

    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "lr": []
    }

    best_val_loss = float("inf")
    os.makedirs(save_checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_checkpoint_dir, "custom_multilingual_transformer.pt")

    # 5. Training Loop
    print("\n[Phase 3] Starting Training Loop...")
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for src, tgt in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: input is tgt[:, :-1], label is tgt[:, 1:]
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)

            # Reshape for cross entropy
            loss = criterion(
                output.contiguous().view(-1, tokenizer.vocab_size),
                tgt_expected.contiguous().view(-1)
            )
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            current_lr = scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                output = model(src, tgt_input)
                loss = criterion(
                    output.contiguous().view(-1, tokenizer.vocab_size),
                    tgt_expected.contiguous().view(-1)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "num_layers": num_layers,
                    "d_ff": d_ff,
                    "dropout": dropout,
                    "pad_idx": tokenizer.pad_id,
                    "share_embeddings": True
                },
                "tokenizer_path": tokenizer_path,
                "languages": ["en", "hi", "ta", "te"],
                "best_val_loss": best_val_loss,
                "history": history,
                "epoch": epoch,
                "seed": seed,
                "torch_version": str(torch.__version__)
            }, checkpoint_path)

    total_time = round(time.time() - start_train_time, 2)
    print(f"\nTraining completed in {total_time}s! Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")

    # Save training history JSON
    history_path = "ml_lab/evaluation/results/custom_multilingual_training_history.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "checkpoint_path": checkpoint_path,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "history": history
    }


if __name__ == "__main__":
    train_multilingual_model(epochs=60)
