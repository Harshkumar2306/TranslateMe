# TranslateMe: Comprehensive Deep Technical Audit & Rebuild Roadmap

**Date**: November 22, 2026  
**Auditor**: Senior Deep Learning Research & Systems Engineer  
**Objective**: Comprehensive line-by-line inspection of the current repository, identifying all pretrained dependencies, architectural defects, misleading metrics, and formulating the blueprint for a 100% genuine, from-scratch Multilingual Transformer Neural Machine Translation (NMT) system.

---

## 1. Executive Summary & Brutal Truth

### The Brutal Reality:
The repository in its current state is **fundamentally a pretrained model wrapper with an educational toy script attached**, not an authentic custom-trained multilingual translation system:

1. **Pretrained Model Dependency**: 
   - 95% of actual translation functionality relies on Meta AI's **`facebook/nllb-200-distilled-600M`** (a 615-million parameter pretrained model downloaded from Hugging Face).
   - On backend startup (`backend/app/main.py`), the server loads 2.46 GB of pretrained NLLB weights into memory.
   - The 200-language selector in the UI is simply NLLB's Flores-200 vocabulary.

2. **The "Custom Transformer" Fallacy**:
   - While the PyTorch modules in `ml_lab/custom_transformer/` correctly write out the tensor math for Multi-Head Attention and Encoder-Decoder layers following Vaswani et al. (2017), the model has **never been trained as a real translation model**.
   - It was trained on only **35 hardcoded sentence pairs** using naive Python `.split()` whitespace tokenization.
   - Its entire vocabulary is **56 English words** and **57 Hindi words**. If you type any word outside those 35 sentences, it generates `<unk>`.
   - It is **bilingual only (English $\rightarrow$ Hindi)** and has zero multilingual capacity (no Tamil, no Telugu, no target-language control tokens).

3. **Fabricated / Misleading Evaluation Metrics**:
   - In `ml_lab/evaluation/evaluate.py`, the Custom Transformer is evaluated on the exact 3 sentences it was trained on:
     ```python
     sources = ["hello how are you", "attention is all you need", "machine translation connects the world"]
     references = ["namaste aap kaise hain", "attention hi sab kuch hai", "machine anuvaad duniya ko jodta hai"]
     ```
   - This produced **100.0 BLEU** and **100.0 chrF++**, which was then saved into `ml_lab/evaluation/results/evaluation_results.json` and presented in charts and documentation as an empirical evaluation result. Evaluating on the training set is a severe scientific defect.

4. **Hardcoded Deep Learning Lab Visualizations**:
   - In `frontend/src/pages/DeepLearningLabPage.tsx`, the self-attention heatmap is a hardcoded static 6x6 float array (`[0.92, 0.02, 0.01...]`), not extracted dynamically from the model during an actual inference pass.

---

## 2. Granular Line-by-Line Inspection by Component

### A. `ml_lab/custom_transformer/` (The Architecture)
- **Strengths**:
  - The implementation is clean, modular PyTorch: `attention.py`, `encoder.py`, `decoder.py`, `embeddings.py`, `feed_forward.py`, and `sublayers.py`.
  - Scaled dot-product attention correctly computes $\text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
  - Positional encoding correctly computes sinusoidal functions.
  - Sublayer connections support both Pre-LN and Post-LN.
- **Defects & Limitations**:
  - **No Beam Search**: `transformer.py` only implements basic `greedy_decode()`. NMT requires true beam search decoding with length penalties and candidate tracking.
  - **No Multilingual Conditioning**: There are no target language prefix tokens (e.g., `<2hi>`, `<2ta>`, `<2te>`, `<2en>`) to guide the decoder which language to generate.
  - **Attention Weight Propagation Missing**: During decoding, multi-head attention weights are stored on internal module attributes (`self.last_attn_weights`), but not returned to the caller or API for visualization.
  - **Separate Vocabularies**: Embeddings use separate source and target vocabularies rather than a unified shared multilingual embedding table with tied generator weights (standard in multilingual NMT).

### B. `ml_lab/training/` (The Training Pipeline)
- **Defects**:
  - `train_nllb.py`: A fine-tuning script built for Meta AI's NLLB-200. It depends entirely on pretrained weights.
  - `train_custom_transformer.py`:
    - Uses `SimpleVocab` (word-level splitting with `.split()`). This fails on Indian languages (Tamil, Telugu, Hindi) where subwords and Unicode conjuncts require Byte-Pair Encoding (BPE) or SentencePiece.
    - Trains on a fixed batch of 35 sentences in memory.
    - No validation set evaluation or early stopping.
    - No learning rate warmup or gradient accumulation for larger batch training.
  - `dataset_loader.py`: Contains 36 hardcoded sentences. No loaders for real open parallel corpora (like Samanantar, OPUS, or CCAligned).

### C. `ml_lab/evaluation/` (Evaluation Suite)
- **Defects**:
  - `evaluate.py`: Evaluates NLLB on 5 sentences, and evaluates the custom transformer on 3 sentences from the training data.
  - Generates `evaluation_results.json` claiming 100.0 BLEU for the custom transformer.
  - Does not evaluate on an independent held-out test split.
  - Lacks per-language-pair evaluation across English, Hindi, Tamil, and Telugu.

### D. `backend/app/services/` & `backend/app/api/`
- **Defects**:
  - `backend/app/main.py`: The `lifespan` handler automatically downloads and loads NLLB-200 on every single server start.
  - `backend/app/services/nllb_service.py`: 262 lines of wrapper code for NLLB-200.
  - `backend/app/services/language_codes.py`: Hardcoded 200 NLLB Flores language codes.
  - `backend/app/api/routes.py`: Translation route originally passed all requests to NLLB.

### E. Frontend (`frontend/src/`)
- **Defects**:
  - `TranslatorPage.tsx`: Lists 200 languages that the custom model cannot translate.
  - `DeepLearningLabPage.tsx`: Visualizes fake attention weights from a static constant array.
  - `ModelInfoPage.tsx`: Documents Meta AI's NLLB-200 model card instead of the custom Transformer.

### F. Documentation (`README.md` & `docs/`)
- **Defects**:
  - The documentation claims to be a universal 200-language translation engine when only the pretrained Hugging Face model supports that.
  - Claims 100.0 BLEU score for the custom model.

---

## 3. Required Architectural Changes & Rebuild Plan

To make this genuinely **YOUR OWN TRANSFORMER FROM SCRATCH**:

```
+------------------------------------------------------------------------------------------------+
|                                TRANSLATEME: FROM-SCRATCH REBUILD ARCHITECTURE                  |
+------------------------------------------------------------------------------------------------+
|                                                                                                |
|   1. REAL DATA PIPELINE:                                                                       |
|      - Clean, aligned parallel corpora for English, Hindi, Tamil, Telugu (OPUS / Samanantar)   |
|      - Strict train (80%), validation (10%), and held-out test (10%) splits                     |
|                                                                                                |
|   2. MULTILINGUAL SUBWORD TOKENIZER:                                                           |
|      - Trained from scratch on our parallel text (SentencePiece / BPE, 16K-32K vocab)          |
|      - Special tokens: <pad>, <sos>, <eos>, <unk>, <2en>, <2hi>, <2ta>, <2te>                  |
|                                                                                                |
|   3. CUSTOM PYTORCH TRANSFORMER (Random Initialization):                                       |
|      - Encoder (6 layers, 8 heads, d_model=512, d_ff=2048, dropout=0.1)                       |
|      - Decoder (6 layers, 8 heads, d_model=512, d_ff=2048, dropout=0.1)                       |
|      - Shared Embedding Table + Tied Output Projection (Vaswani et al. 2017)                   |
|      - Autoregressive Generation with Greedy Decoding & Beam Search (k=4)                      |
|      - Dynamic Cross-Attention weight extraction for live heatmap rendering                    |
|                                                                                                |
|   4. GENUINE TRAINING LOOP:                                                                    |
|      - AdamW optimizer, Noam scheduler (linear warmup + inverse sqrt decay)                   |
|      - Label smoothing cross-entropy with padding ignore                                       |
|      - Validation loss tracking & best checkpoint selection                                    |
|                                                                                                |
|   5. SCIENTIFIC EVALUATION:                                                                    |
|      - True held-out test set evaluation using SacreBLEU and chrF++                            |
|      - Zero fabricated scores                                                                  |
|                                                                                                |
|   6. PURE CUSTOM APPLICATION:                                                                  |
|      - Remove NLLB background preloading completely                                            |
|      - Translation engine runs strictly on the trained custom checkpoint                       |
|      - UI displays only supported languages (English, Hindi, Tamil, Telugu)                   |
|      - Deep Learning Lab visualizes real live multi-head attention weights                     |
+------------------------------------------------------------------------------------------------+
```

---

## 4. Phased Execution Roadmap

1. **Phase 1: Architecture Completion**
   - Enhance `ml_lab/custom_transformer/`:
     - Implement Shared Embeddings with tied softmax weights.
     - Implement Beam Search decoding with length normalization.
     - Implement Attention Extraction during decoding.
     - Add configurable model scaling: Small (smoke test), Medium, Base (6L-8H-512D).

2. **Phase 2: Multilingual Tokenizer Training**
   - Train a SentencePiece / BPE multilingual tokenizer from scratch on our multilingual corpus.
   - Embed language direction control tokens: `<2en>`, `<2hi>`, `<2ta>`, `<2te>`.
   - Ensure full Unicode script integrity for Devanagari, Tamil, and Telugu scripts.

3. **Phase 3: Parallel Data Pipeline**
   - Implement data loading, cleaning, Unicode NFC normalization, deduplication, length filtering, and stratified splitting.
   - Provide a verifiable parallel corpus across English, Hindi, Tamil, and Telugu.

4. **Phase 4: Training Engine & Checkpointing**
   - Build a production-grade PyTorch training loop with AdamW, Noam learning rate schedule, gradient clipping, validation loss evaluation, and automatic checkpoint saving.
   - Train on Apple MPS / GPU.

5. **Phase 5: Genuine Evaluation**
   - Evaluate on a held-out test split using SacreBLEU and chrF++.
   - Report honest, unvarnished translation scores.

6. **Phase 6: Full-Stack Integration**
   - Remove NLLB preloading from backend startup.
   - Wire the custom Transformer as the sole translation service.
   - Connect live attention weights to the Deep Learning Lab.
