# Dataset Engineering & Preprocessing Documentation

This document describes the data architecture, preprocessing stages, language distribution, and record-level split isolation implemented in `ml_lab/datasets/`.

---

## 1. Supported Languages & Script Coverage

The TranslateMe parallel corpus covers four core languages:

| Language | ISO-639 | Script | Unicode Block | Character Range |
| :--- | :---: | :--- | :--- | :--- |
| **English** | `en` | Latin | Basic Latin & Supplements | `U+0020 - U+007E` |
| **Hindi** | `hi` | Devanagari | Devanagari Block | `U+0900 - U+097F` |
| **Tamil** | `ta` | Tamil | Tamil Block | `U+0B80 - U+0BFF` |
| **Telugu** | `te` | Telugu | Telugu Block | `U+0C00 - U+0C7F` |

---

## 2. Multi-Stage Preprocessing Pipeline

Implemented in `ml_lab/datasets/data_pipeline.py`:

```
Raw Parallel Corpus
        │
        ▼
[ Stage 1: Unicode NFC Normalization ] ────> Composes split vowels/consonants
        │
        ▼
[ Stage 2: Whitespace & Control Sanitization ] ─> Strips control characters, normalizes spaces
        │
        ▼
[ Stage 3: Length & Ratio Validation ] ────> Words ∈ [1, 80], max length ratio ≤ 2.5
        │
        ▼
[ Stage 4: Deduplication ] ────────────────> Hashing prevents identical sample pairs
        │
        ▼
[ Stage 5: Record-Level Split Partitioning ] ─> 80% Train, 10% Val, 10% Held-Out Test
```

### 1. Unicode NFC Normalization
Indic languages utilize complex vowel signs (*matras*) and conjunct consonants. If encoded in decomposed NFD form, tokenizers misinterpret isolated combining diacritics. NFC normalization ensures consistent, canonical grapheme representation:
```python
text = unicodedata.normalize("NFC", text.strip())
```

### 2. Length Ratio Filtering
Eliminates unaligned or corrupted sentence pairs where one translation is disproportionately longer or shorter than the other:
$$\text{Ratio} = \frac{\max(\text{len}_{\text{src}}, \text{len}_{\text{tgt}})}{\min(\text{len}_{\text{src}}, \text{len}_{\text{tgt}})} \le 2.5$$

---

## 3. Strict Record-Level Split Partitioning

To eliminate **all** data leakage, splits are partitioned at the **aligned record level** before any directional pairs are generated:

| Split | Percentage | Aligned 4-Way Records | Directional Pairs | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | **80%** | 32 Records | 192 Pairs | Backpropagation parameter updates |
| **Validation** | **10%** | 4 Records | 24 Pairs | Checkpoint selection & early stopping |
| **Held-Out Test** | **10%** | 4 Records | 24 Pairs | Final independent SacreBLEU & chrF++ evaluation |
| **Total** | **100%** | **40 Records** | **240 Pairs** | Full Corpus |

### Record-Level Split Isolation & Leakage Verification
1. **No Exact Aligned-Record Overlap**: No sentence in the held-out test split appears in the training or validation partitions ($0.00\%$ exact match).
2. **Directional Expansion Isolation**: Splits are partitioned at the aligned 4-way record level prior to directional pair creation. If English sentence $A$ translates to Hindi sentence $B$ in training, reverse pair $B \rightarrow A$ is strictly confined to the training partition and will never appear in validation or test.
3. **Tokenizer Isolation**: Subword vocabulary merges were trained strictly on the training partition text.
