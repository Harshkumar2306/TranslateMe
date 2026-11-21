"""
Fine-Tuning Pipeline for Pretrained NLLB-200 Multilingual Transformer
Supports Apple MPS, NVIDIA CUDA, and CPU execution.
Configurable learning rate, batch size, epochs, and checkpoint saving.
"""

import os
import time
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from backend.app.core.device import get_device
from training.dataset_loader import load_parallel_corpus, prepare_dataset_splits


class TranslationDataset(Dataset):
    def __init__(self, data_path: str):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_translation_batch(batch, tokenizer, src_lang: str, tgt_lang: str, max_len: int = 128):
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]

    tokenizer.src_lang = src_lang
    model_inputs = tokenizer(sources, max_length=max_len, padding=True, truncation=True, return_tensors="pt")

    # Target forced token
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    labels = tokenizer(targets, max_length=max_len, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100  # PyTorch ignore index

    model_inputs["labels"] = labels
    model_inputs["forced_bos_token_id"] = tgt_token_id
    return model_inputs


def run_finetuning(config_path: str = "training/config.yaml", max_steps: int = None):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model_name = cfg["model"]["pretrained_model_name"]
    src_lang = cfg["model"]["src_lang"]
    tgt_lang = cfg["model"]["tgt_lang"]
    epochs = cfg["training"]["num_train_epochs"]
    batch_size = cfg["training"]["per_device_train_batch_size"]
    lr = float(cfg["training"]["learning_rate"])
    output_dir = cfg["training"]["output_dir"]

    print(f"\n=======================================================")
    print(f"LinguaAI: Fine-Tuning NLLB-200 ({src_lang} -> {tgt_lang})")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"=======================================================\n")

    # 1. Prepare data
    splits = prepare_dataset_splits(load_parallel_corpus("en-hi"))
    train_dataset = TranslationDataset(splits["train"])
    val_dataset = TranslationDataset(splits["val"])

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 3. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_translation_batch(b, tokenizer, src_lang, tgt_lang)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_translation_batch(b, tokenizer, src_lang, tgt_lang)
    )

    # 4. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["training"]["warmup_ratio"]),
        num_training_steps=total_steps
    )

    history = {
        "model": model_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    step_counter = 0
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            step_counter += 1

            if max_steps and step_counter >= max_steps:
                break

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                input_ids = val_batch["input_ids"].to(device)
                attention_mask = val_batch["attention_mask"].to(device)
                labels = val_batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        history["epochs"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Saved new best checkpoint to {output_dir}")

        if max_steps and step_counter >= max_steps:
            print("Reached requested max_steps for testing.")
            break

    # Save training history
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/nllb_finetuning_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nFine-tuning completed successfully!")
    return history


if __name__ == "__main__":
    run_finetuning()
