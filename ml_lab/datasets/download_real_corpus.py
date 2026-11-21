#!/usr/bin/env python3
"""
TranslateMe V2: Power Rebuild - Download Real Parallel Corpora
Downloads high-quality EN-HI parallel data from OPUS (IIT Bombay corpus)
and prepares it for training a production-grade custom Transformer.

This script downloads REAL parallel data — not toy sentences.
"""

import os
import sys
import json
import random
import hashlib
import unicodedata
import re
import gzip
import urllib.request
import zipfile
import io
from pathlib import Path
from typing import List, Tuple, Dict

# ---- Configuration ----
DATA_DIR = Path("ml_lab/datasets/real_corpus")
PROCESSED_DIR = Path("ml_lab/datasets/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# We will download from multiple sources and combine
# Source 1: OPUS IITB (IIT Bombay EN-HI) - ~1.6M parallel sentences
# Source 2: OPUS Tatoeba (small but high quality multi-language)
# Source 3: Our curated corpus (40 4-way records)

MAX_PAIRS_PER_DIRECTION = 150_000  # Limit for 8GB RAM M2 machine
MIN_SRC_LEN = 3
MAX_SRC_LEN = 150  # characters
MIN_TGT_LEN = 3
MAX_TGT_LEN = 200
MAX_RATIO = 3.0  # max len ratio between src and tgt


def download_file(url: str, dest: str, desc: str = ""):
    """Download a file with progress."""
    if os.path.exists(dest):
        print(f"  [CACHED] {desc}: {dest}")
        return
    print(f"  [DOWNLOADING] {desc}: {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [DONE] {size_mb:.1f} MB downloaded to {dest}")
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text for NMT training."""
    if not isinstance(text, str):
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'[\r\t\f\v]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove very long repeated characters (spam)
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)
    return text.strip()


def is_valid_pair(src: str, tgt: str) -> bool:
    """Check if a parallel pair is valid for training."""
    if not src or not tgt:
        return False
    if len(src) < MIN_SRC_LEN or len(tgt) < MIN_TGT_LEN:
        return False
    if len(src) > MAX_SRC_LEN or len(tgt) > MAX_TGT_LEN:
        return False
    
    # Length ratio check
    ratio = len(src) / max(len(tgt), 1)
    if ratio > MAX_RATIO or ratio < (1 / MAX_RATIO):
        return False
    
    # Check it's not just numbers/punctuation
    alpha_src = sum(1 for c in src if c.isalpha())
    alpha_tgt = sum(1 for c in tgt if c.isalpha())
    if alpha_src < 2 or alpha_tgt < 2:
        return False
    
    # Reject if src == tgt (copy, not translation)
    if src.strip().lower() == tgt.strip().lower():
        return False
    
    return True


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Remove duplicate pairs using hash-based dedup."""
    seen = set()
    unique = []
    for src, tgt in pairs:
        key = hashlib.md5(f"{src.strip().lower()}|||{tgt.strip().lower()}".encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append((src, tgt))
    return unique


def download_opus_corpus():
    """
    Download EN-HI parallel data from OPUS.
    Uses direct OPUS download links for the IITB corpus.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: DOWNLOADING REAL PARALLEL CORPORA")
    print("=" * 70)
    
    all_en_hi_pairs = []
    
    # ---- Source 1: Try to download OPUS IITB corpus ----
    print("\n[Source 1] OPUS IIT Bombay EN-HI Corpus")
    iitb_en_path = DATA_DIR / "IITB.en-hi.en.gz"
    iitb_hi_path = DATA_DIR / "IITB.en-hi.hi.gz"
    
    try:
        # OPUS Moses format URLs for IITB
        download_file(
            "https://object.pouta.csc.fi/OPUS-IITB/v1/moses/en-hi.txt.zip",
            str(DATA_DIR / "iitb_en_hi.zip"),
            "IITB EN-HI (OPUS)"
        )
        
        # Extract the zip
        zip_path = DATA_DIR / "iitb_en_hi.zip"
        if zip_path.exists():
            print("  [EXTRACTING] IITB corpus...")
            with zipfile.ZipFile(str(zip_path), 'r') as zf:
                zf.extractall(str(DATA_DIR))
            
            # Read the extracted files
            en_file = DATA_DIR / "IITB.en-hi.en"
            hi_file = DATA_DIR / "IITB.en-hi.hi"
            
            if en_file.exists() and hi_file.exists():
                with open(en_file, 'r', encoding='utf-8', errors='replace') as ef:
                    en_lines = ef.readlines()
                with open(hi_file, 'r', encoding='utf-8', errors='replace') as hf:
                    hi_lines = hf.readlines()
                
                min_len = min(len(en_lines), len(hi_lines))
                print(f"  [LOADED] {min_len:,} raw EN-HI pairs from IITB")
                
                for i in range(min_len):
                    en = clean_text(en_lines[i])
                    hi = clean_text(hi_lines[i])
                    if is_valid_pair(en, hi):
                        all_en_hi_pairs.append((en, hi))
                
                print(f"  [CLEANED] {len(all_en_hi_pairs):,} valid pairs after filtering")
            else:
                # Try to find the files with different names
                for f in DATA_DIR.iterdir():
                    print(f"    Found: {f.name}")
                    
    except Exception as e:
        print(f"  [WARNING] IITB download failed: {e}")
        print("  Falling back to alternative sources...")
    
    # ---- Source 2: Try Tatoeba from OPUS ----
    if len(all_en_hi_pairs) < 10000:
        print("\n[Source 2] OPUS Tatoeba EN-HI Corpus")
        try:
            download_file(
                "https://object.pouta.csc.fi/OPUS-Tatoeba/v2024-07-01/moses/en-hi.txt.zip",
                str(DATA_DIR / "tatoeba_en_hi.zip"),
                "Tatoeba EN-HI"
            )
            
            zip_path = DATA_DIR / "tatoeba_en_hi.zip"
            if zip_path.exists():
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(DATA_DIR / "tatoeba"))
                
                tat_dir = DATA_DIR / "tatoeba"
                en_files = list(tat_dir.glob("*.en"))
                hi_files = list(tat_dir.glob("*.hi"))
                
                if en_files and hi_files:
                    with open(en_files[0], 'r', encoding='utf-8', errors='replace') as ef:
                        en_lines = ef.readlines()
                    with open(hi_files[0], 'r', encoding='utf-8', errors='replace') as hf:
                        hi_lines = hf.readlines()
                    
                    added = 0
                    for i in range(min(len(en_lines), len(hi_lines))):
                        en = clean_text(en_lines[i])
                        hi = clean_text(hi_lines[i])
                        if is_valid_pair(en, hi):
                            all_en_hi_pairs.append((en, hi))
                            added += 1
                    print(f"  [ADDED] {added:,} pairs from Tatoeba")
        except Exception as e:
            print(f"  [WARNING] Tatoeba download failed: {e}")

    # ---- Source 3: Try WikiMatrix ----
    if len(all_en_hi_pairs) < 10000:
        print("\n[Source 3] OPUS WikiMatrix EN-HI")
        try:
            download_file(
                "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-hi.txt.zip",
                str(DATA_DIR / "wikimatrix_en_hi.zip"),
                "WikiMatrix EN-HI"
            )
            
            zip_path = DATA_DIR / "wikimatrix_en_hi.zip"
            if zip_path.exists():
                with zipfile.ZipFile(str(zip_path), 'r') as zf:
                    zf.extractall(str(DATA_DIR / "wikimatrix"))
                
                wm_dir = DATA_DIR / "wikimatrix"
                en_files = list(wm_dir.glob("*.en"))
                hi_files = list(wm_dir.glob("*.hi"))
                
                if en_files and hi_files:
                    with open(en_files[0], 'r', encoding='utf-8', errors='replace') as ef:
                        en_lines = ef.readlines()
                    with open(hi_files[0], 'r', encoding='utf-8', errors='replace') as hf:
                        hi_lines = hf.readlines()
                    
                    added = 0
                    for i in range(min(len(en_lines), len(hi_lines))):
                        en = clean_text(en_lines[i])
                        hi = clean_text(hi_lines[i])
                        if is_valid_pair(en, hi):
                            all_en_hi_pairs.append((en, hi))
                            added += 1
                    print(f"  [ADDED] {added:,} pairs from WikiMatrix")
        except Exception as e:
            print(f"  [WARNING] WikiMatrix download failed: {e}")

    # ---- Source 4: Our curated multilingual corpus ----
    print("\n[Source 4] TranslateMe Curated Multilingual Corpus")
    try:
        from ml_lab.datasets.curated_corpus import generate_multilingual_training_set
        curated = generate_multilingual_training_set()
        curated_added = 0
        for item in curated:
            if item.get("target_lang") == "hi":
                en = clean_text(item["source"])
                hi = clean_text(item["target"])
                if en and hi:
                    all_en_hi_pairs.append((en, hi))
                    curated_added += 1
        print(f"  [ADDED] {curated_added} pairs from curated corpus")
    except Exception as e:
        print(f"  [WARNING] Curated corpus: {e}")
    
    # ---- Deduplicate ----
    print(f"\n[DEDUP] Total raw pairs: {len(all_en_hi_pairs):,}")
    all_en_hi_pairs = deduplicate_pairs(all_en_hi_pairs)
    print(f"[DEDUP] After deduplication: {len(all_en_hi_pairs):,}")
    
    # ---- Subsample if needed for memory constraints ----
    if len(all_en_hi_pairs) > MAX_PAIRS_PER_DIRECTION:
        print(f"[SUBSAMPLE] Reducing to {MAX_PAIRS_PER_DIRECTION:,} pairs for 8GB RAM training")
        random.seed(42)
        random.shuffle(all_en_hi_pairs)
        all_en_hi_pairs = all_en_hi_pairs[:MAX_PAIRS_PER_DIRECTION]
    
    # ---- Split: 90% train, 5% val, 5% test ----
    print(f"\n[SPLIT] Splitting {len(all_en_hi_pairs):,} pairs...")
    random.seed(42)
    random.shuffle(all_en_hi_pairs)
    
    n = len(all_en_hi_pairs)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)
    
    train_pairs = all_en_hi_pairs[:train_end]
    val_pairs = all_en_hi_pairs[train_end:val_end]
    test_pairs = all_en_hi_pairs[val_end:]
    
    print(f"  Train: {len(train_pairs):,}")
    print(f"  Val:   {len(val_pairs):,}")
    print(f"  Test:  {len(test_pairs):,}")
    
    # ---- Write JSONL files (bidirectional: en->hi AND hi->en) ----
    def write_split(pairs, path, include_reverse=True):
        with open(path, 'w', encoding='utf-8') as f:
            for src, tgt in pairs:
                # EN -> HI
                f.write(json.dumps({
                    "source": src,
                    "target": tgt,
                    "source_lang": "en",
                    "target_lang": "hi"
                }, ensure_ascii=False) + "\n")
                
                if include_reverse:
                    # HI -> EN (reverse direction)
                    f.write(json.dumps({
                        "source": tgt,
                        "target": src,
                        "source_lang": "hi",
                        "target_lang": "en"
                    }, ensure_ascii=False) + "\n")
    
    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"
    test_path = PROCESSED_DIR / "test.jsonl"
    
    write_split(train_pairs, train_path)
    write_split(val_pairs, val_path)
    write_split(test_pairs, test_path)
    
    # Count actual lines
    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path))
    test_count = sum(1 for _ in open(test_path))
    
    print(f"\n[WRITTEN]")
    print(f"  {train_path}: {train_count:,} directional pairs")
    print(f"  {val_path}: {val_count:,} directional pairs")
    print(f"  {test_path}: {test_count:,} directional pairs")
    
    # Save statistics
    stats = {
        "total_unique_pairs": len(all_en_hi_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_directional": train_count,
        "val_directional": val_count,
        "test_directional": test_count,
        "directions": ["en->hi", "hi->en"],
        "max_src_len": MAX_SRC_LEN,
        "max_tgt_len": MAX_TGT_LEN,
    }
    with open(PROCESSED_DIR / "split_statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"DATA DOWNLOAD COMPLETE")
    print(f"Total training pairs: {train_count:,} (bidirectional EN<->HI)")
    print(f"{'=' * 70}")
    
    return train_path, val_path, test_path


if __name__ == "__main__":
    download_opus_corpus()
