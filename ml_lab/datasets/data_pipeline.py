"""
Data Preprocessing, Cleaning, Filtering, and Partitioning Pipeline
for Multilingual Neural Machine Translation.
"""

import os
import re
import unicodedata
import random
from typing import List, Tuple, Dict, Any, Optional


def normalize_unicode_text(text: str) -> str:
    """
    Applies Unicode NFC Normalization and strips unprintable characters.
    Critical for Indic scripts (Hindi, Tamil, Telugu) where complex conjuncts
    and vowel diacritics must be in canonical composed representation.
    """
    if not text:
        return ""
    # NFC Normalization
    text = unicodedata.normalize("NFC", text.strip())
    # Remove control characters except standard whitespace
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", " "))
    # Collapse multiple whitespaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def filter_parallel_pair(
    src: str,
    tgt: str,
    min_len_words: int = 1,
    max_len_words: int = 80,
    max_ratio: float = 3.0
) -> bool:
    """
    Applies length, non-empty, and length-ratio validation checks.
    """
    if not src or not tgt:
        return False

    src_words = src.split()
    tgt_words = tgt.split()

    if len(src_words) < min_len_words or len(tgt_words) < min_len_words:
        return False

    if len(src_words) > max_len_words or len(tgt_words) > max_len_words:
        return False

    # Check length ratio between source and target
    ratio = max(len(src_words), len(tgt_words)) / max(1, min(len(src_words), len(tgt_words)))
    if ratio > max_ratio:
        return False

    return True


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Removes duplicate (source, target) sentence pairs preserving order."""
    seen = set()
    deduped = []
    for src, tgt in pairs:
        key = (src, tgt)
        if key not in seen:
            seen.add(key)
            deduped.append((src, tgt))
    return deduped


def clean_and_prepare_parallel_data(
    raw_pairs: List[Tuple[str, str]],
    min_len_words: int = 1,
    max_len_words: int = 80,
    max_ratio: float = 3.0
) -> List[Tuple[str, str]]:
    """
    Executes end-to-end cleaning:
    1. Unicode NFC normalization
    2. Length and alignment filtering
    3. Duplicate pair removal
    """
    cleaned = []
    for s, t in raw_pairs:
        s_norm = normalize_unicode_text(s)
        t_norm = normalize_unicode_text(t)
        if filter_parallel_pair(s_norm, t_norm, min_len_words, max_len_words, max_ratio):
            cleaned.append((s_norm, t_norm))

    return deduplicate_pairs(cleaned)


def split_parallel_dataset(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Partitions parallel dataset into train, validation, and held-out test splits.
    Guarantees that test examples are never leaked into training or validation.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    return {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }


def compute_dataset_stats(pairs: List[Tuple[str, str]], name: str = "Dataset") -> Dict[str, Any]:
    """Computes verifiable descriptive statistics for parallel sentence pairs."""
    if not pairs:
        return {"name": name, "total_pairs": 0}

    src_lengths = [len(s.split()) for s, _ in pairs]
    tgt_lengths = [len(t.split()) for _, t in pairs]

    return {
        "dataset_name": name,
        "total_pairs": len(pairs),
        "src_avg_words": round(sum(src_lengths) / len(src_lengths), 2),
        "tgt_avg_words": round(sum(tgt_lengths) / len(tgt_lengths), 2),
        "src_max_words": max(src_lengths),
        "tgt_max_words": max(tgt_lengths),
        "src_min_words": min(src_lengths),
        "tgt_min_words": min(tgt_lengths),
    }


def save_parallel_split(split_data: List[Tuple[str, str]], filepath: str):
    """Saves TSV file of format: source_text \t target_text."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for s, t in split_data:
            f.write(f"{s}\t{t}\n")


def load_parallel_split(filepath: str) -> List[Tuple[str, str]]:
    """Loads parallel sentence pairs from TSV file."""
    if not os.path.exists(filepath):
        return []
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs
