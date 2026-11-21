"""
Educational Training Script for Custom PyTorch Transformer
Demonstrates training a sequence-to-sequence Transformer from scratch on parallel text pairs.
Follows 'Attention Is All You Need' optimization details (Noam scheduler, Adam optimizer).
"""

import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from ml_lab.custom_transformer import Seq2SeqTransformer


class SimpleVocab:
    """Simple word-level vocabulary builder with special tokens."""
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        self.w2i = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3,
        }
        self.i2w = {idx: w for w, idx in self.w2i.items()}

    def build_vocab(self, texts: List[str]):
        for text in texts:
            words = text.strip().lower().split()
            for w in words:
                if w not in self.w2i:
                    idx = len(self.w2i)
                    self.w2i[w] = idx
                    self.i2w[idx] = w

    def encode(self, text: str) -> List[int]:
        words = text.strip().lower().split()
        return [self.w2i.get(w, self.w2i[self.UNK_TOKEN]) for w in words]

    def decode(self, ids: List[int]) -> str:
        words = []
        for i in ids:
            if i == self.w2i[self.EOS_TOKEN]:
                break
            if i not in (self.w2i[self.PAD_TOKEN], self.w2i[self.SOS_TOKEN]):
                words.append(self.i2w.get(i, self.UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.w2i)


class ParallelDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: SimpleVocab, tgt_vocab: SimpleVocab, max_len: int = 32):
        self.data = []
        for src, tgt in pairs:
            src_ids = [src_vocab.w2i[SimpleVocab.SOS_TOKEN]] + src_vocab.encode(src) + [src_vocab.w2i[SimpleVocab.EOS_TOKEN]]
            tgt_ids = [tgt_vocab.w2i[SimpleVocab.SOS_TOKEN]] + tgt_vocab.encode(tgt) + [tgt_vocab.w2i[SimpleVocab.EOS_TOKEN]]

            # Pad or truncate
            if len(src_ids) > max_len:
                src_ids = src_ids[:max_len]
            if len(tgt_ids) > max_len:
                tgt_ids = tgt_ids[:max_len]

            self.data.append((src_ids, tgt_ids))

        self.max_len = max_len
        self.src_pad = src_vocab.w2i[SimpleVocab.PAD_TOKEN]
        self.tgt_pad = tgt_vocab.w2i[SimpleVocab.PAD_TOKEN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, src_pad: int = 0, tgt_pad: int = 0):
    src_list, tgt_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)

    padded_src = []
    for s in src_list:
        pad_len = max_src - len(s)
        padded_src.append(torch.cat([s, torch.full((pad_len,), src_pad, dtype=torch.long)]))

    padded_tgt = []
    for t in tgt_list:
        pad_len = max_tgt - len(t)
        padded_tgt.append(torch.cat([t, torch.full((pad_len,), tgt_pad, dtype=torch.long)]))

    return torch.stack(padded_src), torch.stack(padded_tgt)


class NoamLRScheduler:
    """
    Learning rate schedule from Section 5.3 of 'Attention Is All You Need':
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 400, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.factor * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


# Sample educational bilingual pairs (English -> Hindi phonetic / transliterated / simple vocabulary)
SAMPLE_PAIRS = [
    ("hello how are you", "namaste aap kaise hain"),
    ("i am learning deep learning", "main deep learning seekh raha hoon"),
    ("transformers are powerful neural networks", "transformers shaktishaali neural network hain"),
    ("attention is all you need", "attention hi sab kuch hai"),
    ("machine translation connects the world", "machine anuvaad duniya ko jodta hai"),
    ("artificial intelligence is the future", "artificial intelligence bhavishya hai"),
    ("good morning my friend", "shubh prabhat mere mitra"),
    ("what is your name", "aapka naam kya hai"),
    ("thank you very much", "aapka bahut bahut dhanyavaad"),
    ("knowledge is power", "gyaan hi shakti hai"),
    ("the model translates text accurately", "model paath ka sahi anuvaad karta hai"),
    ("neural machine translation uses deep learning", "neural machine translation deep learning ka upayog karta hai"),
    ("welcome to lingua ai", "lingua ai mein aapka swaagat hai"),
    ("where do you live", "aap kahan rehte hain"),
    ("have a nice day", "aapka din shubh ho"),
]


def train_custom_model(
    epochs: int = 30,
    d_model: int = 128,
    n_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 256,
    batch_size: int = 4,
    device: str = "cpu"
) -> Dict:
    print(f"--- Training Custom Transformer on {device} ---")

    # 1. Vocabularies
    src_vocab = SimpleVocab()
    tgt_vocab = SimpleVocab()
    src_vocab.build_vocab([p[0] for p in SAMPLE_PAIRS])
    tgt_vocab.build_vocab([p[1] for p in SAMPLE_PAIRS])

    print(f"Source Vocab Size: {len(src_vocab)}, Target Vocab Size: {len(tgt_vocab)}")

    # 2. Dataset & Loader
    dataset = ParallelDataset(SAMPLE_PAIRS, src_vocab, tgt_vocab)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.w2i[SimpleVocab.PAD_TOKEN], tgt_vocab.w2i[SimpleVocab.PAD_TOKEN])
    )

    # 3. Model
    model = Seq2SeqTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        n_heads=n_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1,
        pad_idx=src_vocab.w2i[SimpleVocab.PAD_TOKEN]
    ).to(device)

    # 4. Loss and Optimizer
    pad_idx = tgt_vocab.w2i[SimpleVocab.PAD_TOKEN]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLRScheduler(optimizer, d_model=d_model, warmup_steps=50, factor=0.5)

    history = {"epochs": [], "loss": [], "lr": []}

    # 5. Training Loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Target input: tgt[:, :-1], Target label: tgt[:, 1:]
            tgt_input = tgt_batch[:, :-1]
            tgt_expected = tgt_batch[:, 1:]

            optimizer.zero_grad()
            output = model(src_batch, tgt_input)

            # Reshape for cross entropy
            loss = criterion(output.contiguous().view(-1, len(tgt_vocab)), tgt_expected.contiguous().view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            current_lr = scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history["epochs"].append(epoch)
        history["loss"].append(round(avg_loss, 4))
        history["lr"].append(current_lr)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # 6. Evaluation test on training sample
    model.eval()
    test_src = "hello how are you"
    src_tokens = torch.tensor([[src_vocab.w2i[SimpleVocab.SOS_TOKEN]] + src_vocab.encode(test_src) + [src_vocab.w2i[SimpleVocab.EOS_TOKEN]]], device=device)
    generated = model.greedy_decode(
        src_tokens,
        sos_idx=tgt_vocab.w2i[SimpleVocab.SOS_TOKEN],
        eos_idx=tgt_vocab.w2i[SimpleVocab.EOS_TOKEN],
        max_len=15
    )
    predicted_text = tgt_vocab.decode(generated[0].tolist())
    print(f"\n[Test Inference]")
    print(f"Source:    {test_src}")
    print(f"Generated: {predicted_text}")

    # 7. Save Checkpoint & Results
    os.makedirs("ml_lab/checkpoints", exist_ok=True)
    os.makedirs("ml_lab/evaluation/results", exist_ok=True)

    checkpoint_path = "ml_lab/checkpoints/custom_transformer_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "src_vocab": src_vocab.w2i,
        "tgt_vocab": tgt_vocab.w2i,
        "config": {
            "d_model": d_model,
            "n_heads": n_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
        },
        "history": history
    }, checkpoint_path)
    print(f"Saved custom model checkpoint to: {checkpoint_path}")

    with open("ml_lab/evaluation/results/custom_transformer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_custom_model(epochs=30, device=device)
