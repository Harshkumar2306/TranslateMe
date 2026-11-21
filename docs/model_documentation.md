# Model Documentation: Custom Multilingual Seq2Seq Transformer

## 1. Model Overview

The TranslateMe translation engine is a **from-scratch multilingual Sequence-to-Sequence Transformer** built in pure PyTorch and trained on 4-way parallel Indic-English language corpora.

| Attribute | Specification |
| :--- | :--- |
| **Model Architecture** | Standard Encoder-Decoder Transformer (Vaswani et al., 2017) |
| **Implementation** | `ml_lab/custom_transformer/` (Pure PyTorch modules) |
| **Trainable Parameters** | **6,033,408 (6.03M)** |
| **Pretrained Weights** | **Zero (Random Gaussian Initialization $\mathcal{N}(0, 1)$)** |
| **Vocabulary Size** | 2,000 Byte-Pair Encoded (BPE) subwords |
| **Supported Languages** | English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`) |
| **Embedding Dimension ($d_{\text{model}}$)** | 256 |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 1024 |
| **Encoder Layers ($N_{\text{enc}}$)** | 3 |
| **Decoder Layers ($N_{\text{dec}}$)** | 3 |
| **Attention Heads ($h$)** | 4 ($d_k = d_v = 64$) |
| **Dropout Rate** | 0.1 |
| **Weight Tying** | Shared input subword embeddings & output linear generator |
| **Target Conditioning** | Special control tokens: `<2en>`, `<2hi>`, `<2ta>`, `<2te>` |
| **Inference Decoding** | Autoregressive Beam Search ($k=1$ to $4$) + Repetition Penalty |

---

## 2. Parameter Calculation Breakdown

The model's **6,033,408 parameters** are allocated across layers as follows:

1. **Shared Token Embedding & Projection**:
   $$|V| \times d_{\text{model}} = 2000 \times 256 = 512,000 \text{ weights}$$
   Output generator bias: $2,000$ weights.

2. **Encoder Stack (3 Layers)**:
   - Multi-Head Self-Attention ($W_Q, W_K, W_V, W_O$):
     $$4 \times (256 \times 256 + 256) = 263,168 \text{ per layer}$$
   - Position-Wise Feed-Forward Network ($W_1, b_1, W_2, b_2$):
     $$(256 \times 1024 + 1024) + (1024 \times 256 + 256) = 525,568 \text{ per layer}$$
   - Two Layer Normalization layers:
     $$2 \times (2 \times 256) = 1,024 \text{ per layer}$$
   - Total per Encoder layer: $\approx 789,760 \text{ weights} \times 3 = 2,369,280 \text{ weights}$.

3. **Decoder Stack (3 Layers)**:
   - Masked Self-Attention: $263,168 \text{ per layer}$
   - Encoder-Decoder Cross-Attention: $263,168 \text{ per layer}$
   - Feed-Forward Network: $525,568 \text{ per layer}$
   - Three Layer Normalization layers: $1,536 \text{ per layer}$
   - Total per Decoder layer: $\approx 1,053,440 \text{ weights} \times 3 = 3,160,320 \text{ weights}$.

4. **Final Layer Normalization**:
   $$2 \times 256 = 512 \text{ weights}$$

$$\text{Total Trainable Parameters} = 6,033,408$$

---

## 3. Supported Languages & Control Tokens

| Language | ISO Code | Native Name | Script | Direction Token |
| :--- | :---: | :--- | :--- | :---: |
| English | `en` | English | Latin | `<2en>` |
| Hindi | `hi` | हिन्दी | Devanagari | `<2hi>` |
| Tamil | `ta` | தமிழ் | Tamil | `<2ta>` |
| Telugu | `te` | తెలుగు | Telugu | `<2te>` |

The target language control token is prepended to the decoder input sequence immediately after the `sos_id` token during both training (teacher forcing) and inference (autoregressive generation).

---

## 4. Software Versions & Reproducibility

- PyTorch: `2.6.0`
- Tokenizers: `0.21.0`
- SacreBLEU: `2.6.0`
- Random Seed: `42`
- Checkpoint: `ml_lab/checkpoints/custom_multilingual_transformer.pt`
