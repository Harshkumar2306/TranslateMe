# TranslateMe — From-Scratch Multilingual Neural Machine Translation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 35 Passed](https://img.shields.io/badge/Tests-35%20Passed-brightgreen.svg)](tests/)

> **A 100% Genuine From-Scratch Multilingual Neural Machine Translation System.**  
> Built as an advanced Deep Learning research system featuring an independently implemented PyTorch Sequence-to-Sequence Transformer initialized randomly and trained through our own training pipeline across English, Hindi, Tamil, and Telugu. Zero pretrained translation weights (no NLLB, no HuggingFace, no mBART).

---

## Key Achievements & Guarantees

1. **Pure PyTorch Architecture**: Implemented entirely from scratch in `ml_lab/custom_transformer/` without using `torch.nn.Transformer` or Hugging Face model classes.
2. **Production 27.97M Parameter Model**: Trained across 10 epochs on Kaggle Dual Tesla T4 GPUs on 183,314 parallel pairs with PyTorch AMP FP16, reaching a validation loss of **2.2957**.
3. **Modern SOTA 100M+ Architecture (`TranslateMe-V2`)**:
   - **Pre-LN with RMSNorm**: Zero mean-centering overhead; provides direct gradient highway for deep networks.
   - **Rotary Position Embeddings (`RoPE`)**: Relative distance-aware token rotations for cross-lingual reordering.
   - **SwiGLU Gated Feed-Forward**: Bilinear gated activations replacing basic ReLU.
   - **QK-Norm & FlashAttention-2**: RMSNorm on queries & keys to eliminate logit explosion, coupled with PyTorch fused `scaled_dot_product_attention`.
   - **111.7M Parameter Configuration**: Optimized 6-layer Encoder + 6-layer Decoder architecture ($d_{\text{model}}=768, d_{\text{ff}}=2048, 12\text{ heads}$).
4. **Resumable Training Pipeline with Pause Feature**:
   - **Touch-File Pause**: Gracefully pauses training when `pause.flag` is detected, writing `checkpoint_latest.pt` without losing compute.
   - **Signal Traps**: Catches `SIGINT` (Ctrl+C) and `SIGTERM` to safely checkpoint.
   - **Deterministic Resume**: Restores model weights, AdamW 1st & 2nd moments, AMP GradScaler state, LR scheduler step, and CPU/CUDA RNG states.
5. **ByteLevel Multilingual Subword Tokenizer**: A shared 8,192 Byte-Pair Encoding (BPE) subword vocabulary trained from scratch across Latin, Devanagari, Tamil, and Telugu scripts, complete with direction control tokens (`<2en>`, `<2hi>`, `<2ta>`, `<2te>`).
6. **Scientifically Honest Evaluation**: 35/35 passing unit tests covering architecture, causal masking, future token leakage, tokenizer roundtrips, and API endpoints.

---

## Architectural Specifications

### 1. Production Active Checkpoint (28M)
* **Model Checkpoint**: [`ml_lab/checkpoints/custom_multilingual_transformer.pt`](ml_lab/checkpoints/custom_multilingual_transformer.pt) (66.17 MB FP16)
* **Total Parameters**: **27,974,144 (27.97M)**
* **Encoder / Decoder Layers**: **6 Layers Encoder + 6 Layers Decoder** ($N = 12$ sub-stacks)
* **Attention Heads**: **6 Heads** ($d_k = d_v = 64$)
* **Hidden Dimension ($d_{\text{model}}$)**: **384**
* **Feed-Forward Dimension ($d_{\text{ff}}$)**: **1536**
* **Vocabulary Size**: 8,192 ByteLevel BPE multilingual subwords
* **Hardware Execution**: Apple Silicon GPU (`mps`) / NVIDIA CUDA / CPU

### 2. Modern SOTA 100M+ Architecture
* **Implementation**: [`ml_lab/custom_transformer/modern_transformer.py`](ml_lab/custom_transformer/modern_transformer.py)
* **Blueprint**: [`docs/MODERN_100M_TRANSFORMER_BLUEPRINT.md`](docs/MODERN_100M_TRANSFORMER_BLUEPRINT.md)
* **Total Parameters**: **111,700,224 (111.7M)**
* **Encoder / Decoder Layers**: 6 Encoder + 6 Decoder (or asymmetric 8 Enc / 6 Dec for 125.8M)
* **Attention Heads**: **12 Heads** ($d_k = d_v = 64$) with QK-Norm and RoPE
* **Hidden Dimension ($d_{\text{model}}$)**: **768**
* **Feed-Forward Dimension ($d_{\text{ff}}$)**: **2048 (SwiGLU)**
* **Vocabulary Size**: 16,384 multilingual subwords

---

## Directory Structure

```
TranslateMe/
├── backend/app/                  # FastAPI Application Service
│   ├── api/routes.py             # Translation, Attention & Analytics Endpoints
│   ├── services/
│   │   └── custom_multilingual_service.py # PyTorch Inference & Live Attention Service
│   └── main.py                   # Server Lifecycle & Production Static Mounter
│
├── frontend/                     # React 18 + TypeScript + Vite UI
│   ├── src/pages/
│   │   ├── TranslatorPage.tsx    # Primary Multilingual Translator Interface
│   │   ├── DeepLearningLabPage.tsx # Live Cross-Attention Heatmaps & Loss Curves
│   │   └── ModelInfoPage.tsx     # Architecture Specifications
│   └── dist/                     # Compiled Production Assets
│
├── ml_lab/                       # Deep Learning Research Lab
│   ├── custom_transformer/       # Pure PyTorch Transformer Implementation
│   │   ├── modern_transformer.py # SOTA 100M Seq2Seq Transformer (RoPE, RMSNorm, SwiGLU, QK-Norm)
│   │   ├── modern_attention.py   # Multi-Head Attention with QK-Norm and FlashAttention-2 SDPA
│   │   ├── modern_layers.py      # Modern Pre-LN Encoder & Decoder Layers
│   │   ├── rms_norm.py           # Root Mean Square Layer Normalization
│   │   ├── rope.py               # Rotary Position Embedding
│   │   ├── swiglu.py             # Swish Gated Linear Unit
│   │   └── transformer.py        # Classical 28M Seq2SeqTransformer
│   ├── tokenizer/                # Multilingual BPE Tokenizer (From Scratch)
│   │   ├── language_registry.py  # Language Config & Control Tokens (<2en>, <2hi>, ...)
│   │   └── multilingual_tokenizer.py # ByteLevel BPE Subword Trainer & Serializer
│   ├── training/                 # Model Training Pipeline
│   │   └── resumable_trainer.py  # Production Resumable Trainer with Pause & Checkpoint Recovery
│   ├── kaggle/                   # Cloud GPU Training Pipelines
│   │   ├── translateme_kaggle_trainer.py   # Kaggle Dual Tesla T4 Trainer Script
│   │   └── translateme_kaggle_trainer.ipynb # Kaggle Training Notebook
│   └── checkpoints/              # Trained PyTorch Model Checkpoints (.pt)
│
├── tests/                        # Comprehensive Unit & Architecture Test Suite (35 Tests)
│   ├── test_modern_transformer.py # Modern SOTA RoPE, RMSNorm, SwiGLU, & Resumable Trainer Tests
│   ├── test_custom_transformer.py # Masking, Attention & Shape Invariance Tests
│   ├── test_multilingual_tokenizer.py # Subword Encoding/Decoding Tests
│   └── test_api.py               # REST API Endpoint Tests
│
└── docs/                         # In-Depth Research Documentation
    ├── MODERN_100M_TRANSFORMER_BLUEPRINT.md # SOTA 100M Transformer Blueprint & Data Strategy
    ├── CURRENT_PROJECT_AUDIT.md  # Comprehensive Codebase Audit
    ├── CUSTOM_TRANSFORMER_MATHEMATICS.md # Equations, LayerNorm, Loss Formulations
    └── REPRODUCIBILITY.md        # Step-by-Step Reproduction Guide
```

---

## Quickstart & Verification

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Comprehensive Architecture Test Suite
```bash
source .venv/bin/activate
pytest tests/ -v
```
*(All 35 tests pass: SOTA RoPE, RMSNorm, SwiGLU, QK-Norm, Resumable Trainer, causal masking, tokenizer, and API endpoints).*

### 3. Start Application Server
```bash
source .venv/bin/activate
python backend/app/main.py
```
Open **`http://localhost:8000`** to access the live web application.

---

## License

This project is licensed under the **MIT License**. All Transformer architectures are independent academic implementations built from scratch with PyTorch.
