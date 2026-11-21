# Empirical Experiments & Architecture Ablations

This document logs the architecture ablations, decoding strategy comparisons, and optimization experiments conducted on the TranslateMe Custom Transformer.

---

## 1. Architecture Configurations Compared

| Configuration | Layers (Enc/Dec) | Heads | $d_{\text{model}}$ | $d_{\text{ff}}$ | Vocab | Parameters | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Toy Baseline (Prior)** | 2 / 2 | 2 | 64 | 256 | 56 words | 100K | Evaluated on train set (fake 100 BLEU) |
| **Research Small** | 3 / 3 | 4 | 256 | 1024 | 2,000 subwords | **6,033,408** | **Selected Production Baseline** |
| **Medium Baseline** | 6 / 6 | 8 | 512 | 2048 | 8,000 subwords | 48,245,760 | Configured in `TransformerConfig.base()` |

---

## 2. Decoding Strategy Ablations

### Greedy Decoding vs Beam Search ($k=4$)
Across the 6 evaluation directions on the held-out test set:

| Decoding Strategy | Average Latency | Avg chrF++ | Repetition Tendency |
| :--- | :---: | :---: | :--- |
| **Greedy Search ($k=1$)** | **0.42s** | 4.85 | Low (with $\gamma = 1.2$ repetition penalty) |
| **Beam Search ($k=2$)** | **0.85s** | 5.32 | Very low with early stopping |
| **Beam Search ($k=4$)** | **1.45s** | **5.82** | Lowest error rate with length penalty 0.6 |

---

## 3. Embedding Sharing Ablation

- **Separate Embeddings**:
  - Encoder input embedding: $|V| \times d_{\text{model}} = 512,000$
  - Decoder input embedding: $|V| \times d_{\text{model}} = 512,000$
  - Output linear projection: $|V| \times d_{\text{model}} = 512,000$
  - Total parameter overhead: $1,536,000$
- **Shared Tied Embeddings (Press & Wolf, 2017)**:
  - Total embedding weights: $512,000$
  - Result: 1.02M fewer parameters, faster convergence, and cross-lingual subword alignment.
