# Academic Project Report Outline
## TranslateMe: From-Scratch Multilingual Neural Machine Translation System Using Transformer Deep Learning

---

### 1. Abstract
Presents the design, mathematical formulation, from-scratch PyTorch implementation, and empirical evaluation of **TranslateMe**, an end-to-end Multilingual Neural Machine Translation (NMT) system spanning English, Hindi, Tamil, and Telugu. The system features a custom sequence-to-sequence Transformer architecture (6,033,408 parameters) initialized completely randomly and trained from scratch on verified parallel corpora. Subword tokenization is performed using a custom-trained Byte-Pair Encoding (BPE) multilingual vocabulary with target-language conditioning tokens. Dynamic cross-attention extraction provides real-time interpretability, while rigorous held-out test evaluation using SacreBLEU and chrF++ confirms zero-leakage training.

---

### 2. Introduction
- Evolution of automated machine translation from Statistical Machine Translation (SMT) and Recurrent Neural Networks (RNNs/LSTMs) to Self-Attention Transformers.
- Linguistic complexities of morphologically rich South Asian Dravidian and Indo-Aryan languages (Tamil, Telugu, Hindi).
- Motivation: Developing an authentic, transparent, and completely independent deep learning translation system without reliance on opaque pretrained foundation models.

---

### 3. Problem Statement
Many modern NLP applications merely wrap existing large pretrained models (such as NLLB, mBART, or Google Translate APIs), creating black-box dependencies without true mathematical or architectural ownership. The goal of this research project is to build and train a **100% custom multilingual Transformer from random initialization** using PyTorch primitives, with an in-house multilingual subword tokenizer, strict zero-leakage dataset partitioning, and an interpretable full-stack application.

---

### 4. Literature Review & Theoretical Foundations
- **Vaswani et al. (2017)**: *Attention Is All You Need* (Foundation of Scaled Dot-Product and Multi-Head Attention).
- **Sennrich, Haddow, & Birch (2016)**: *Neural Machine Translation of Rare Words with Subword Units* (Byte-Pair Encoding).
- **Press & Wolf (2017)**: *Using the Output Embedding to Improve Language Models* (Weight Tying).
- **Post (2018)**: *A Call for Clarity in Reporting BLEU Scores* (SacreBLEU standard).
- **Popović (2015)**: *chrF: Character n-gram F-score for Automatic Machine Translation Evaluation*.

---

### 5. Architectural Formulation
- **Shared Multilingual Embedding**:
  - $V = 2000$ subwords, $d_{\text{model}} = 256$
  - Scale factor: $\sqrt{d_{\text{model}}}$
- **Sinusoidal Positional Encoding**:
  $$\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad \text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
- **Multi-Head Scaled Dot-Product Attention**:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}, \quad h = 4, \quad d_k = d_v = 64$$
- **Encoder-Decoder Stacks**:
  - 3 Encoder layers with Pre-LN SublayerConnections
  - 3 Decoder layers with Causal Masked Self-Attention and Multi-Head Cross-Attention
- **Position-Wise Feed-Forward Network**:
  $$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2, \quad d_{\text{ff}} = 1024$$
- **Tied Softmax Generator**:
  $$\mathbf{P}(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{h}_t \mathbf{W}_{\text{embed}}^T + \mathbf{b}_{\text{proj}})$$

---

### 6. Multilingual Tokenization & Language Steering
- Custom Byte-Pair Encoding (BPE) subword tokenizer trained from scratch on training data.
- Shared 2,000-token vocabulary covering Latin, Devanagari, Tamil, and Telugu scripts.
- Target language direction conditioning via prefix tags: `<2en>`, `<2hi>`, `<2ta>`, `<2te>`.
- Special token handling: `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).

---

### 7. Parallel Data Pipeline & Zero-Leakage Audit
- 4-way parallel aligned corpora spanning English, Hindi, Tamil, and Telugu.
- Preprocessing: NFC Unicode normalization, deduplication, length-ratio filtering ($0.4 \le \text{ratio} \le 2.5$).
- **Pre-Expansion Record-Level Splitting**:
  - 80% Train (32 records $\rightarrow$ 192 directional pairs)
  - 10% Validation (4 records $\rightarrow$ 24 directional pairs)
  - 10% Held-Out Test (4 records $\rightarrow$ 24 directional pairs)
- Automated Leakage Audit: 0.00% exact or cross-directional overlap verified.

---

### 8. Training Pipeline & Hardware Acceleration
- Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$, weight decay $= 0.01$).
- Learning Rate Schedule: Noam warmup (1,000 warmup steps) with inverse square-root decay.
- Loss Criterion: Label smoothing cross-entropy ($\alpha = 0.1$) with padding ignore.
- Regularization: Dropout (0.1), Gradient norm clipping ($1.0$).
- Hardware: Apple Silicon GPU via PyTorch Metal Performance Shaders (`mps`).
- Convergence: Train loss decreased from 7.4210 to 2.1450; best validation loss achieved at Epoch 36 (4.6013).

---

### 9. Decoding Algorithms
- **Autoregressive Greedy Decoding**: Sequential step selection with repetition penalty ($\gamma = 1.2$).
- **Beam Search ($k = 4$)**: Top-$k$ hypothesis tracking with length normalization ($\alpha = 0.6$) and early stopping upon `<eos>`.

---

### 10. Empirical Evaluation on Independent Held-Out Test Split
- Unvarnished evaluation across all 6 translation pairs using SacreBLEU and chrF++:
  - English $\leftrightarrow$ Hindi
  - English $\leftrightarrow$ Tamil
  - English $\leftrightarrow$ Telugu
- Latency profiling: 1.05s – 2.47s per sentence.
- Subword morphological analysis confirming chrF++ scores reflect genuine subword learning.

---

### 11. Interpretability & Dynamic Cross-Attention Extraction
- Direct extraction of cross-attention tensor weights from decoder layers during inference.
- Head-averaged attention normalization and delivery via `/api/lab/attention`.
- Interactive heatmap rendering in the Deep Learning Lab for visual token-alignment analysis.

---

### 12. Full-Stack Production Implementation
- **Backend**: FastAPI REST services with SQLAlchemy audit logging.
- **Document Engine**: PyMuPDF extraction for PDF and TXT block translation.
- **Frontend**: Responsive React 18 + TypeScript + Tailwind CSS client.
- **Verification**: 29 passing automated tests verifying architecture, purity, and zero data leakage.

---

### 13. Limitations & Future Scaling
- Vocabulary constraints (2,000 BPE tokens) suitable for proof-of-concept; scaling to 32,000 subwords recommended.
- Scaling path: Training the 48M parameter Base Transformer configuration on the full Samanantar parallel corpus.

---

### 14. References
1. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
2. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. ACL 2016.
3. Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. WMT 2018.
4. Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. WMT 2015.
