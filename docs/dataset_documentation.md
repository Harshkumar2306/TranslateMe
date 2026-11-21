# Dataset Preparation & Preprocessing Pipeline

## 1. Overview
LinguaAI includes a dataset curation, cleaning, and partitioning pipeline in `training/` designed to process parallel bilingual corpora.

## 2. Preprocessing & Sanitization Pipeline
Text cleaning logic is implemented in `training/preprocessing.py`:
- **Unicode NFC Normalization**: Unifies combining characters into standard precomposed glyphs, critical for Indian languages with complex vowel mats and conjuncts.
- **Deduplication**: Removes identical source-target sentence pairs.
- **Empty & Whitespace Sanitization**: Strips excessive carriage returns, tabs, and unprintable characters while keeping punctuation.
- **Length Ratio Filtering**: Eliminates pairs where word count ratios exceed 3.5:1, avoiding truncated or misaligned sentences.

## 3. Dataset Splits
Parallel pairs are automatically partitioned into standard splits:
- **Train Split (80%)**: For model gradient updates
- **Validation Split (10%)**: For checkpoint selection and early stopping
- **Test Split (10%)**: Held out for final SacreBLEU and chrF++ evaluations

All processed files are stored in `datasets/processed/` as clean JSONL files (`train.jsonl`, `val.jsonl`, `test.jsonl`), accompanied by detailed metadata in `dataset_stats.json`.
