# Model Card: TranslateMe Multilingual Neural Machine Translation

## 1. Model Details
- **Developer**: TranslateMe Academic Deep Learning Research Project
- **Languages Supported**: English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`) across all **12 translation directions** (`en↔hi↔ta↔te`)
- **Vocabulary**: **24,576** Byte-Pair Encoded (BPE) subwords with byte-level fallback, Unicode NFC normalization, and target direction control tokens (`<2en>`, `<2hi>`, `<2ta>`, `<2te>`)
- **Weights Source**: Initialized randomly from scratch ($\mathcal{N}(0, 1/\sqrt{d_{\text{model}}})$ embeddings & depth-scaled Xavier uniform linear weights). Zero pretrained weights (no NLLB, no HuggingFace, no mBART).
- **License**: MIT Open Source License

---

## 2. Active Model Architecture (`ModernSeq2SeqTransformer` — Option 1 Asymmetric)

- **Checkpoint Location**: `ml_lab/checkpoints/custom_multilingual_transformer.pt` (`432.3 MB` FP32)
- **Implementation**: `ml_lab/custom_transformer/modern_transformer.py`
- **Model Dimensions**: $d_{\text{model}} = 768$, $N_{\text{enc}} = 8$, $N_{\text{dec}} = 4$, $h = 12$ ($d_k = 64$), $d_{\text{ff}} = 2048$ (SwiGLU)
- **Total Trainable Parameters**: **113,292,800 (113.29 Million)**
- **Training Checkpoint Progress**: **5 epochs / 137,500 optimizer steps**, best validation cross-entropy loss **1.0593**
- **Key Enhancements**:
  1. **Pre-LN with RMSNorm**: Direct residual gradient highways without mean-centering overhead.
  2. **Rotary Position Embeddings (`RoPE`)**: Relative token rotations applied to self-attention queries and keys.
  3. **SwiGLU Gated Feed-Forward**: Bilinear gated activations ($3 \times 768 \times 2048$ projection matrices).
  4. **QK-Norm**: RMSNorm on query and key projections prior to dot-product to eliminate attention logit spikes.
  5. **PyTorch SDPA & KV-Cache**: Fused scaled dot-product attention plus $O(L)$ KV-cached autoregressive greedy and batched beam search decoding.
  6. **LayerScale & Gemma-2 Logit Soft-Capping**: Residual scaling ($\gamma = 0.1$) and logit soft-capping at $[-30.0, +30.0]$, paired with 3-way tied embeddings (`shared_embedding` = `lm_head`).

*(Legacy baseline checkpoints — 27.97M classical 6E/6D — are archived separately under `ml_lab/checkpoints/legacy_28m/`.)*

---

## 3. Training & Evaluation Methodology
- **Record-Level Partitioning**: Cluster-level partitioning across aligned parallel records prior to 12-direction expansion (`ml_lab/datasets/build_100m_dataset.py`).
- **Optimization**: AdamW (`weight_decay=0.01`, exempting 1D RMSNorm/LayerScale/bias parameters), linear warmup + cosine annealing LR schedule, and label smoothing ($\epsilon=0.1$).
- **Inference Safeguards**: Target script vocabulary masking (`get_script_mask`) and 3-gram repetition blocking during decoding.

---

## 4. Empirical Evaluation (12-Direction Held-Out Suite)

Evaluated via `ml_lab/evaluation/evaluate.py` on the independent 4-way held-out test records (4 records × 12 directions = 48 evaluations, 0% training overlap) using Beam Search ($k=2$):

- **Mean SacreBLEU**: **10.26** (up to **34.22** on `hi → en`)
- **Mean chrF++**: **43.71** (up to **65.19** on `hi → en`)
- **Empty / Repetition Rate**: **0.0%** across all 12 directions

Full per-direction scores and sample translations are stored in `ml_lab/evaluation/results/custom_multilingual_evaluation_report.json` and `ml_lab/evaluation/results/custom_multilingual_evaluation_report.md`.
