"""
Data Cleaning and Text Preprocessing for Multilingual Parallel Corpora
Preserves Indic scripts (Devanagari, Tamil, Telugu, etc.) while sanitizing malformed tokens.
"""

import re
import unicodedata
from typing import List, Tuple, Dict


def normalize_unicode(text: str) -> str:
    """Normalizes Unicode characters into NFC form to standardize Indic diacritics."""
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Cleans raw text while strictly preserving Indian scripts, Latin accents, and valid punctuation.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    # Remove HTML tags if present
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace multiple whitespaces and control characters with a single space
    text = re.sub(r'[\r\t\f\v]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def filter_parallel_pairs(
    pairs: List[Tuple[str, str]],
    min_len: int = 1,
    max_len: int = 256,
    max_ratio: float = 3.5
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Filters and cleans parallel sentence pairs.
    Deduplicates, checks length boundaries, and ratio checks to filter misaligned pairs.
    """
    cleaned_pairs = []
    seen = set()
    stats = {
        "total_initial": len(pairs),
        "removed_empty": 0,
        "removed_duplicates": 0,
        "removed_length_outlier": 0,
        "removed_ratio_outlier": 0,
        "retained": 0
    }

    for src, tgt in pairs:
        c_src = clean_text(src)
        c_tgt = clean_text(tgt)

        if not c_src or not c_tgt:
            stats["removed_empty"] += 1
            continue

        pair_key = (c_src, c_tgt)
        if pair_key in seen:
            stats["removed_duplicates"] += 1
            continue
        seen.add(pair_key)

        src_words = len(c_src.split())
        tgt_words = len(c_tgt.split())

        if src_words < min_len or tgt_words < min_len or src_words > max_len or tgt_words > max_len:
            stats["removed_length_outlier"] += 1
            continue

        ratio = max(src_words, tgt_words) / max(1, min(src_words, tgt_words))
        if ratio > max_ratio and max(src_words, tgt_words) > 6:
            stats["removed_ratio_outlier"] += 1
            continue

        cleaned_pairs.append((c_src, c_tgt))

    stats["retained"] = len(cleaned_pairs)
    return cleaned_pairs, stats
