# Model Card: TranslateMe Multilingual Neural Machine Translation

## 1. Model Details
- **Developer**: TranslateMe Academic Deep Learning Research Project
- **Languages Supported**: English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`) across all 6 directional pairs
- **Vocabulary**: 8,192 Byte-Pair Encoded (BPE) subwords with byte-level fallback and Unicode NFC normalization
- **Weights Source**: Initialized randomly from scratch ($\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ embeddings & Xavier uniform linear weights). Zero pretrained weights (no NLLB, no HuggingFace, no mBART).
- **License**: MIT Open Source License

---

## 2. Model Architectures & Configurations

TranslateMe maintains two distinct model tiers:

### Tier 1: Canonical Trained Baseline (Active Checkpoint)
- **Checkpoint Location**: `ml_lab/checkpoints/custom_multilingual_transformer.pt` (66.17 MB FP16)
- **Architecture**: Classic Encoder-Decoder Transformer (Vaswani et al., 2017)
- **Model Dimensions**: $d_{\text{model}} = 384$, $N_{\text{enc}} = 6$, $N_{\text{dec}} = 6$, $h = 6$, $d_{\text{ff}} = 1536$
- **Total Trainable Parameters**: **27,974,144 (27.97 Million)**
- **Positional Encoding**: Sinusoidal fixed positional encoding
- **Weight Tying**: Three-way shared embeddings (Encoder Embedding = Decoder Embedding = Projection Matrix)
- **Validation Loss**: **2.2957** (trained on Kaggle Dual Tesla T4 GPUs with PyTorch AMP FP16)

### Tier 2: Modern 100M+ Architecture (`TranslateMe-V2`)
- **Source Implementation**: `ml_lab/custom_transformer/modern_transformer.py`
- **Model Dimensions**: $d_{\text{model}} = 768$, $N_{\text{enc}} = 6$, $N_{\text{dec}} = 6$, $h = 12$, $d_{\text{ff}} = 2048$ (SwiGLU)
- **Total Trainable Parameters**: **111,723,264 (111.72 Million)**
- **Key Enhancements**:
  1. **Pre-LN with RMSNorm**: Infinite gradient highway without mean-centering overhead.
  2. **Rotary Position Embeddings (`RoPE`)**: Relative token rotations applied to self-attention queries and keys.
  3. **SwiGLU Gated Feed-Forward**: Bilinear gated activations ($3 \times 768 \times 2048$ projection matrices).
  4. **QK-Norm**: RMSNorm on query and key projections prior to dot-product to eliminate logit explosion.
  5. **PyTorch SDPA**: Scaled dot-product attention utilizing fused backends.
  6. **Gemma 2 Logit Soft-Capping**: Bounds decoder logits strictly to $[-30.0, +30.0]$ preventing runaway repetition.

---

## 3. Training & Evaluation Methodology
- **Record-Level Partitioning**: Strict separation at the aligned 4-way record level prior to directional expansion. No exact aligned-record overlap between train, val, and test partitions.
- **Optimization**: AdamW with weight decay exemption for 1D parameters (RMSNorm, LayerScale, biases), Noam warmup schedule, and label smoothing ($\epsilon=0.1$).
- **Gradient Accumulation**: Dynamic loss scaling for partial microbatch groups to eliminate gradient distortion.
- **Validation Loss**: Token-weighted cross-entropy loss normalized by valid non-padding target tokens:
  $$\mathcal{L}_{\text{val}} = \frac{\sum \text{token losses}}{\sum \text{valid tokens}}$$

---

## 4. Empirical Evaluation & Observed Behavior

### Held-Out Test Metrics
- **English → Tamil**: SacreBLEU: **0.86**, chrF++: **11.17**
- **English → Hindi**: SacreBLEU: **0.40**, chrF++: **3.31**
- **English → Telugu**: SacreBLEU: **0.00**, chrF++: **6.06**
- **Hindi / Tamil / Telugu → English**: SacreBLEU: **0.00**, chrF++: **4.54 – 6.84**

### Known Limitations & Error Analysis
1. **Data Famine on Colloquial Vocabulary**: When tested on everyday sentences outside the curated corpus (e.g. *"ram eats a mango but he can eat apple"*), the model exhibits domain shift and semantic hallucination due to limited exposure during seed training.
2. **Proper Nouns & Acronyms**: Technical terms and acronyms require larger vocabulary coverage ($32,000$ subwords) and full-scale training on the AI4Bharat Samanantar dataset (~150,000+ pairs per direction) to generalize effectively.
3. **Interpretability**: Attention heatmaps provide educational visualization of cross-attention alignment, but do not replace empirical translation evaluation metrics.
