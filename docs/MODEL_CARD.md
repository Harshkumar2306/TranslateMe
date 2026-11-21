# Model Card: TranslateMe Custom Multilingual Transformer

## Model Details
- **Developer**: TranslateMe Academic Deep Learning Research Project
- **Model Architecture**: Encoder-Decoder Seq2Seq Transformer (Vaswani et al., 2017)
- **Model Dimensions**: $d_{\text{model}} = 256$, $N_{\text{enc}} = 3$, $N_{\text{dec}} = 3$, $h = 4$, $d_{\text{ff}} = 1024$
- **Total Parameters**: 6,033,408 (6.03 Million)
- **Weights Source**: Random Gaussian $\mathcal{N}(0, 1)$ & Xavier uniform initialization (Zero pretrained weights)
- **Languages Supported**: English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`)
- **Vocabulary**: 2,000 Byte-Pair Encoded (BPE) subwords with byte fallback
- **License**: MIT Open Source License

---

## Intended Use
- **Primary Use**: Educational and research exploration of multilingual Neural Machine Translation architectures implemented in pure PyTorch.
- **Out of Scope**: High-stakes production or medical/legal translation without human verification.

---

## Training Data & Methodology
- **Data Source**: Aligned 4-way parallel Indic-English sentences processed with Unicode NFC normalization.
- **Split Protocol**: Strict record-level 80% Train, 10% Validation, 10% Held-Out Test.
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.98, \text{decay}=0.01$) with Noam learning rate schedule (50 warmup steps).
- **Loss**: Label-smoothed cross-entropy ($\epsilon=0.1$) ignoring padding tokens.

---

## Evaluation Results (Held-Out Test Set)
- **English → Tamil**: SacreBLEU: 0.58, chrF++: 8.08
- **English → Telugu**: SacreBLEU: 0.00, chrF++: 4.15
- **English → Hindi**: SacreBLEU: 0.00, chrF++: 2.58
- **Average Latency**: 1.05s – 2.47s per sentence on Apple Silicon MPS

---

## Limitations & Ethical Considerations
1. Low-resource seed training corpus results in limited vocabulary coverage for unseen domain entities.
2. Character n-gram scores (chrF++) indicate morphological grounding, but complex syntactic word reordering requires scaling to large-scale corpora.
