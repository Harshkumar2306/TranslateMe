"""
Unit tests for Data Pipeline, Cleaning, and Partitioning
"""

import pytest
from ml_lab.datasets import (
    normalize_unicode_text,
    filter_parallel_pair,
    deduplicate_pairs,
    split_parallel_dataset,
    compute_dataset_stats,
    generate_parallel_pairs,
    get_all_monolingual_sentences
)


def test_unicode_normalization():
    # Combining characters normalized to composed form
    sample = "नमस्ते\u200B  world  "
    normalized = normalize_unicode_text(sample)
    assert "  " not in normalized
    assert normalized.startswith("नमस्ते")


def test_filtering_and_deduplication():
    # Length filtering
    assert filter_parallel_pair("hello world", "नमस्ते दुनिया")
    assert not filter_parallel_pair("", "empty source")
    assert not filter_parallel_pair("one word", "this target sentence is way too long to match a single word ratio", max_ratio=2.0)

    # Deduplication
    pairs = [("hello", "hi"), ("world", "duniya"), ("hello", "hi")]
    deduped = deduplicate_pairs(pairs)
    assert len(deduped) == 2


def test_split_parallel_dataset():
    dummy = [(f"src_{i}", f"tgt_{i}") for i in range(100)]
    splits = split_parallel_dataset(dummy, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    # Ensure zero data leakage between train and test
    train_set = set(splits["train"])
    test_set = set(splits["test"])
    assert len(train_set.intersection(test_set)) == 0


def test_curated_multilingual_corpus():
    en_hi = generate_parallel_pairs("en", "hi")
    en_ta = generate_parallel_pairs("en", "ta")
    en_te = generate_parallel_pairs("en", "te")

    assert len(en_hi) >= 20
    assert len(en_ta) >= 20
    assert len(en_te) >= 20

    monolingual = get_all_monolingual_sentences()
    assert len(monolingual) > 50
