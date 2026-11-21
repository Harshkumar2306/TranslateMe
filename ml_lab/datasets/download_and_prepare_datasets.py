"""
Reproducible Dataset Download, Preprocessing, and Split Generation
Supports legitimate public parallel corpora (OPUS / Samanantar / Curated Suite)
with Unicode NFC normalization, deduplication, length-ratio filtering,
and leakage-free record-level partitioning.
"""

import os
import sys
import json
import hashlib
import unicodedata
from typing import List, Tuple, Dict, Any

from ml_lab.datasets.data_pipeline import (
    normalize_unicode_text,
    filter_parallel_pair,
    deduplicate_pairs
)
from ml_lab.datasets.curated_corpus import (
    MULTILINGUAL_CORPUS_RECORDS,
    get_training_pairs,
    get_validation_pairs,
    get_held_out_test_pairs
)


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare_curated_splits(output_dir: str = "ml_lab/datasets/processed") -> Dict[str, Any]:
    """
    Partitions the 4-way aligned multilingual records into strictly isolated
    Train (80%), Validation (10%), and Test (10%) splits.
    Ensures zero leakage across language directions by splitting records BEFORE expansion.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_pairs = get_training_pairs()
    val_pairs = get_validation_pairs()
    test_pairs = get_held_out_test_pairs()

    # Save split JSONL files
    train_file = os.path.join(output_dir, "train.jsonl")
    val_file = os.path.join(output_dir, "val.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_pairs:
            src, tgt, tgt_lang = item[0], item[1], item[2]
            f.write(json.dumps({"source": src, "target": tgt, "target_lang": tgt_lang}, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_pairs:
            src, tgt, tgt_lang = item[0], item[1], item[2]
            f.write(json.dumps({"source": src, "target": tgt, "target_lang": tgt_lang}, ensure_ascii=False) + "\n")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_pairs:
            src, tgt, tgt_lang = item[0], item[1], item[2]
            f.write(json.dumps({"source": src, "target": tgt, "target_lang": tgt_lang}, ensure_ascii=False) + "\n")

    stats = {
        "train_pairs_count": len(train_pairs),
        "val_pairs_count": len(val_pairs),
        "test_pairs_count": len(test_pairs),
        "train_sha256": compute_sha256(train_file),
        "val_sha256": compute_sha256(val_file),
        "test_sha256": compute_sha256(test_file)
    }

    stats_file = os.path.join(output_dir, "split_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Data Preparation Complete!")
    print(f"  Train Pairs: {len(train_pairs)} (SHA-256: {stats['train_sha256'][:12]}...)")
    print(f"  Val Pairs:   {len(val_pairs)} (SHA-256: {stats['val_sha256'][:12]}...)")
    print(f"  Test Pairs:  {len(test_pairs)} (SHA-256: {stats['test_sha256'][:12]}...)")
    return stats


if __name__ == "__main__":
    prepare_curated_splits()
