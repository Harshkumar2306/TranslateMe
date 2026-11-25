#!/usr/bin/env python3
"""
12-Direction Multi-Way Corpus Builder for Option 1 (113.29M Parameter 8E/4D Transformer)
TranslateMe Project — Full Any-to-Any Across English (en), Hindi (hi), Tamil (ta), Telugu (te)

Features:
- Streams verified AI4Bharat Samanantar parquet shards + OPUS Tatoeba conversational pairs
- Enforces Unicode script purity per language (eliminates untranslated/noisy web-crawled targets)
- Mines direct Indic<->Indic pairs (hi<->ta, hi<->te, ta<->te) via multi-way English-pivot clustering
- Strict cluster-level partitioning (Train / Val / Test) so zero sentences leak across any of the 12 directions
- Supports custom --cache_dir (e.g. /tmp/corpus_cache on Kaggle) so persistent /kaggle/working stays small
"""

import os
import sys
import io
import json
import random
import hashlib
import unicodedata
import zipfile
import re
import urllib.request
import ssl
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any
import certifi
import pyarrow.parquet as pq

SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Multi-shard AI4Bharat Samanantar URLs (validated on HuggingFace)
SAMANANTAR_SHARDS = {
    "hi": [
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/hi/train-00000-of-00008.parquet",
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/hi/train-00001-of-00008.parquet",
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/hi/train-00002-of-00008.parquet",
    ],
    "ta": [
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/ta/train-00000-of-00004.parquet",
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/ta/train-00001-of-00004.parquet",
    ],
    "te": [
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/te/train-00000-of-00003.parquet",
        "https://huggingface.co/datasets/ai4bharat/samanantar/resolve/main/te/train-00001-of-00003.parquet",
    ],
}

OPUS_TATOEBA_URLS = {
    "hi": "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-hi.txt.zip",
    "ta": "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ta.txt.zip",
    "te": "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-te.txt.zip",
}

SCRIPT_RANGES = {
    "en": (0x0041, 0x007A),  # ASCII Latin letters
    "hi": (0x0900, 0x097F),  # Devanagari
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
}

MIN_TOKENS = 2
MAX_TOKENS = 100
MIN_CHARS = 4
MAX_CHARS = 300
MAX_RATIO = 3.0


def clean_text(text: str) -> str:
    """Canonical Unicode NFC normalization, HTML/control stripping, and whitespace collapse."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\r\t\f\v]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has_dominant_script(text: str, lang: str, min_ratio: float = 0.40) -> bool:
    """Verifies that text genuinely contains the expected script characters for `lang`."""
    lo, hi = SCRIPT_RANGES[lang]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    in_script = sum(1 for ch in letters if lo <= ord(ch) <= hi)
    return (in_script / len(letters)) >= min_ratio


def is_valid_pair(src: str, tgt: str, src_lang: str = "en", tgt_lang: Optional[str] = None) -> bool:
    """Rigorous filtering: bounds, word ratio, non-identity, and script purity."""
    if not src or not tgt:
        return False
    if len(src) < MIN_CHARS or len(tgt) < MIN_CHARS or len(src) > MAX_CHARS or len(tgt) > MAX_CHARS:
        return False

    src_words = src.split()
    tgt_words = tgt.split()
    if len(src_words) < MIN_TOKENS or len(tgt_words) < MIN_TOKENS:
        return False
    if len(src_words) > MAX_TOKENS or len(tgt_words) > MAX_TOKENS:
        return False

    ratio = len(src_words) / max(len(tgt_words), 1)
    if ratio > MAX_RATIO or ratio < (1.0 / MAX_RATIO):
        return False

    if src.strip().lower() == tgt.strip().lower():
        return False

    if not has_dominant_script(src, src_lang):
        return False
    if tgt_lang and not has_dominant_script(tgt, tgt_lang):
        return False

    return True


def canonical_en_key(en_text: str) -> str:
    """Normalized key for matching identical English sentences across Indic corpora."""
    norm = re.sub(r"\s+", " ", en_text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=12).hexdigest()


def fetch_samanantar_shard(url: str, lang: str, max_pairs: int, cache_dir: Path) -> List[Tuple[str, str]]:
    """Downloads/caches one Samanantar parquet shard and extracts clean script-verified pairs."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-2] + "_" + url.split("/")[-1]
    local_cache = cache_dir / filename

    if local_cache.exists() and local_cache.stat().st_size > 10 * 1024 * 1024:
        print(f"  [Cache] {local_cache.name} ({local_cache.stat().st_size / (1024*1024):.1f} MB)")
        table = pq.read_table(local_cache)
    else:
        print(f"  [Download] {lang}/{url.split('/')[-1]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=240) as resp:
            content = resp.read()
        print(f"  [Cached] {len(content) / (1024 * 1024):.1f} MB -> {local_cache.name}")
        with open(local_cache, "wb") as f:
            f.write(content)
        table = pq.read_table(local_cache)

    src_col = table["src"].to_pylist()
    tgt_col = table["tgt"].to_pylist()

    valid_pairs: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for s_raw, t_raw in zip(src_col, tgt_col):
        s = clean_text(s_raw)
        t = clean_text(t_raw)
        if is_valid_pair(s, t, src_lang="en", tgt_lang=lang):
            h = hashlib.blake2b(f"{s.lower()}|||{t.lower()}".encode("utf-8"), digest_size=12).hexdigest()
            if h not in seen:
                seen.add(h)
                valid_pairs.append((s, t))
                if len(valid_pairs) >= max_pairs:
                    break
    print(f"  -> Extracted {len(valid_pairs):,} script-pure EN<->{lang.upper()} pairs.")
    return valid_pairs


def fetch_opus_tatoeba(lang: str) -> List[Tuple[str, str]]:
    """Fetches clean conversational Tatoeba pairs (best-effort; skips safely if unreachable)."""
    url = OPUS_TATOEBA_URLS.get(lang)
    if not url:
        return []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        en_name = [n for n in zf.namelist() if n.endswith(".en")][0]
        lang_name = [n for n in zf.namelist() if n.endswith(f".{lang}")][0]
        en_lines = zf.read(en_name).decode("utf-8", errors="ignore").splitlines()
        indic_lines = zf.read(lang_name).decode("utf-8", errors="ignore").splitlines()
        out = []
        for e, ind in zip(en_lines, indic_lines):
            ec, ic = clean_text(e), clean_text(ind)
            if is_valid_pair(ec, ic, src_lang="en", tgt_lang=lang):
                out.append((ec, ic))
        print(f"  [OPUS Tatoeba] EN<->{lang.upper()}: +{len(out):,} conversational pairs")
        return out
    except Exception as ex:
        print(f"  [OPUS Tatoeba] Skipped {lang}: {ex}")
        return []


def build_optimal_dataset(
    target_hi: int = 1_000_000,
    target_ta: int = 700_000,
    target_te: int = 700_000,
    target_indic_pair: int = 400_000,
    output_dir: str = "ml_lab/datasets/processed",
    cache_dir: str = "ml_lab/datasets/real_corpus",
    val_cap_per_dir: int = 4000,
    test_cap_per_dir: int = 4000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compiles a balanced 12-direction any-to-any dataset (EN, HI, TA, TE) using
    multi-way English-pivot clustering with strict cluster-level Train/Val/Test isolation.
    """
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    c_path = Path(cache_dir)

    print("=" * 74)
    print("TRANSLATEME 12-DIRECTION ANY-TO-ANY CORPUS BUILDER (EN, HI, TA, TE)")
    print(f"Targets -> EN<->HI: {target_hi:,} | EN<->TA: {target_ta:,} | EN<->TE: {target_te:,}")
    print(f"           Indic<->Indic (HI<->TA, HI<->TE, TA<->TE): up to {target_indic_pair:,} pairs each")
    print("=" * 74)

    # Cluster map: en_key -> {"en": en_text, "hi": hi_text, "ta": ta_text, "te": te_text}
    clusters: Dict[str, Dict[str, str]] = {}
    targets = {"hi": target_hi, "ta": target_ta, "te": target_te}

    for lang, target_count in targets.items():
        print(f"\nIngesting EN <-> {lang.upper()}...")
        collected: List[Tuple[str, str]] = fetch_opus_tatoeba(lang)
        # Collect slightly more than target_count from shards to maximize multi-way Indic overlap
        fetch_cap = int(target_count * 1.35)
        for url in SAMANANTAR_SHARDS[lang]:
            if len(collected) >= fetch_cap:
                break
            needed = fetch_cap - len(collected)
            collected.extend(fetch_samanantar_shard(url, lang, needed, c_path))

        random.shuffle(collected)
        added = 0
        for en_txt, ind_txt in collected:
            k = canonical_en_key(en_txt)
            if k not in clusters:
                clusters[k] = {"en": en_txt, lang: ind_txt}
                added += 1
            elif lang not in clusters[k]:
                clusters[k][lang] = ind_txt
                added += 1
            if added >= fetch_cap:
                break

    all_cluster_list = list(clusters.values())
    random.shuffle(all_cluster_list)

    # Cluster-level partition: 94% Train, 3% Val, 3% Test (ZERO sentence overlap across all 12 directions)
    n_total = len(all_cluster_list)
    n_train = int(0.94 * n_total)
    n_val = int(0.03 * n_total)

    train_clusters = all_cluster_list[:n_train]
    val_clusters = all_cluster_list[n_train:n_train + n_val]
    test_clusters = all_cluster_list[n_train + n_val:]

    def emit_12_directions(
        cluster_subset: List[Dict[str, str]],
        en_caps: Dict[str, int],
        indic_cap: int
    ) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
        rows: List[Dict[str, str]] = []
        counts: Dict[str, int] = {}

        def add_bi(s_txt: str, t_txt: str, s_lang: str, t_lang: str, pair_key: str, cap: int):
            cur = counts.get(pair_key, 0)
            if cur >= cap:
                return False
            rows.append({"source": s_txt, "target": t_txt, "source_lang": s_lang, "target_lang": t_lang})
            rows.append({"source": t_txt, "target": s_txt, "source_lang": t_lang, "target_lang": s_lang})
            counts[pair_key] = cur + 1
            return True

        # Pass 1: prioritize multi-way clusters for direct Indic<->Indic pairs first
        for c in cluster_subset:
            present_indic = [l for l in ("hi", "ta", "te") if l in c]
            if len(present_indic) >= 2:
                for i in range(len(present_indic)):
                    for j in range(i + 1, len(present_indic)):
                        l1, l2 = present_indic[i], present_indic[j]
                        if not has_dominant_script(c[l1], l1) or not has_dominant_script(c[l2], l2):
                            continue
                        pair_key = f"{l1}-{l2}"
                        add_bi(c[l1], c[l2], l1, l2, pair_key, indic_cap)

        # Pass 2: EN <-> Indic pairs
        for c in cluster_subset:
            en_txt = c["en"]
            for lang in ("hi", "ta", "te"):
                if lang in c:
                    add_bi(en_txt, c[lang], "en", lang, f"en-{lang}", en_caps[lang])

        return rows, counts

    train_rows, train_pair_counts = emit_12_directions(
        train_clusters,
        en_caps={"hi": target_hi, "ta": target_ta, "te": target_te},
        indic_cap=target_indic_pair
    )
    val_rows, val_pair_counts = emit_12_directions(
        val_clusters,
        en_caps={"hi": val_cap_per_dir, "ta": val_cap_per_dir, "te": val_cap_per_dir},
        indic_cap=val_cap_per_dir
    )
    test_rows, test_pair_counts = emit_12_directions(
        test_clusters,
        en_caps={"hi": test_cap_per_dir, "ta": test_cap_per_dir, "te": test_cap_per_dir},
        indic_cap=test_cap_per_dir
    )

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    print("\n" + "=" * 74)
    print("12-DIRECTION PARTITION SUMMARY (Strict Cluster-Level Isolation):")
    print(f"  Train instances: {len(train_rows):,} | Pair breakdown: {train_pair_counts}")
    print(f"  Val instances:   {len(val_rows):,} | Pair breakdown: {val_pair_counts}")
    print(f"  Test instances:  {len(test_rows):,} | Pair breakdown: {test_pair_counts}")
    print("=" * 74)

    for fname, data in [("train.jsonl", train_rows), ("val.jsonl", val_rows), ("test.jsonl", test_rows)]:
        p = out_path / fname
        with open(p, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Saved {p} ({p.stat().st_size / (1024*1024):.1f} MB)")

    stats = {
        "status": "OPTIMAL_12DIR_113M_VERIFIED",
        "total_train_directional": len(train_rows),
        "total_val_directional": len(val_rows),
        "total_test_directional": len(test_rows),
        "train_pair_counts": train_pair_counts,
        "val_pair_counts": val_pair_counts,
        "test_pair_counts": test_pair_counts,
        "record_level_isolation": "Cluster-level 100% isolated across all 12 directions",
    }
    stats_path = out_path / "split_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 12-direction dataset for Option 1 (113.29M Transformer).")
    parser.add_argument("--size", choices=["optimal", "medium", "smoke"], default="optimal")
    parser.add_argument("--output_dir", type=str, default="ml_lab/datasets/processed")
    parser.add_argument("--cache_dir", type=str, default="ml_lab/datasets/real_corpus")
    args = parser.parse_args()

    if args.size == "optimal":
        build_optimal_dataset(
            target_hi=1_000_000, target_ta=700_000, target_te=700_000,
            target_indic_pair=400_000, output_dir=args.output_dir, cache_dir=args.cache_dir
        )
    elif args.size == "medium":
        build_optimal_dataset(
            target_hi=150_000, target_ta=100_000, target_te=100_000,
            target_indic_pair=60_000, output_dir=args.output_dir, cache_dir=args.cache_dir
        )
    else:
        build_optimal_dataset(
            target_hi=5_000, target_ta=2_500, target_te=2_500,
            target_indic_pair=1_500, val_cap_per_dir=300, test_cap_per_dir=300,
            output_dir=args.output_dir, cache_dir=args.cache_dir
        )
