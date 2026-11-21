# Dataset Preparation & Multilingual Data Pipeline

## 1. Overview
The TranslateMe dataset pipeline (`ml_lab/datasets/`) processes 4-way parallel sentences spanning English, Hindi, Tamil, and Telugu. The pipeline is designed for strict data isolation, zero data leakage between splits, and Unicode normalization across diverse Indic scripts.

---

## 2. Multilingual Corpus Architecture

The corpus contains parallel tuples $(s_{\text{en}}, s_{\text{hi}}, s_{\text{ta}}, s_{\text{te}})$ covering general conversation, greetings, technology, science, artificial intelligence, education, and daily communication:

| Language | Script | Character Range | Example Sentence |
| :--- | :--- | :--- | :--- |
| **English** (`en`) | Latin | `U+0020 - U+007E` | *Artificial intelligence is transforming language translation.* |
| **Hindi** (`hi`) | Devanagari | `U+0900 - U+097F` | *आर्टिफिशियल इंटेलिजेंस भाषा अनुवाद को बदल रहा है।* |
| **Tamil** (`ta`) | Tamil | `U+0B80 - U+0BFF` | *செயற்கை நுண்ணறிவு மொழிபெயர்ப்பை மாற்றியமைக்கிறது.* |
| **Telugu** (`te`) | Telugu | `U+0C00 - U+0C7F` | *కృత్రిమ మేధస్సు భాషా అనువాదాన్ని మారుస్తోంది.* |

---

## 3. Preprocessing and Cleaning Stages

Implemented in `ml_lab/datasets/data_pipeline.py`:

1. **Unicode NFC Normalization**:
   Transforms decomposed unicode graphemes (e.g., base consonant + combining matra) into canonical precomposed characters:
   ```python
   text = unicodedata.normalize("NFC", text.strip())
   ```
2. **Whitespace and Character Sanitization**:
   Removes invisible control codes, BOM headers, and collapses multiple spaces while preserving script-specific punctuation (`।`, `.`, `?`).
3. **Length and Ratio Filtering**:
   Discards pairs where character length is $< 3$ or $> 256$, or where length ratio $\frac{\max(L_1, L_2)}{\min(L_1, L_2)} > 2.5$.
4. **Deduplication**:
   Hashes normalized source-target pairs to ensure zero duplicate samples.

---

## 4. Strict Split Partitioning & Leakage Prevention

- **Train Split (80%)**: Used solely during the 40-epoch parameter optimization loop.
- **Validation Split (10%)**: Evaluated at the end of each epoch to determine checkpoint saving.
- **Held-Out Test Split (10%)**: Stored independently in `HELD_OUT_TEST_SUITE` and used exclusively for post-training SacreBLEU and chrF++ benchmarks.

**Leakage Protection:**
Tokenizer training corpus is strictly decoupled from the held-out test split, ensuring that test sentences represent unseen sequences.
