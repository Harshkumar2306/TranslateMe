# TranslateMe — From-Scratch Multilingual Neural Machine Translation

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A 100% Genuine From-Scratch Multilingual Neural Machine Translation System.**  
> Built as an advanced Deep Learning research system featuring an independently implemented PyTorch Sequence-to-Sequence Transformer initialized randomly and trained through our own training pipeline across English, Hindi, Tamil, and Telugu. Zero pretrained translation weights.

---

## Key Achievements & Guarantees

1. **Pure PyTorch Architecture**: Implemented entirely from scratch in `ml_lab/custom_transformer/` without using `torch.nn.Transformer` or Hugging Face model classes.
2. **Zero Pretrained Weights**: Every weight matrix in the 6,033,408 (6.03M) parameter model was initialized randomly from Gaussian / Xavier distributions and trained through our own optimization loop.
3. **Custom Multilingual Subword Tokenizer**: A shared 2,000 Byte-Pair Encoding (BPE) subword vocabulary trained from scratch across Latin, Devanagari, Tamil, and Telugu scripts, complete with direction control tokens (`<2en>`, `<2hi>`, `<2ta>`, `<2te>`).
4. **Isolated Data Pipeline**: Preprocessed 4-way parallel sentences with Unicode NFC normalization, deduplication, length filtering, and strict 80% train / 10% val / 10% held-out test splits.
5. **Scientifically Honest Evaluation**: Removed all invalid 100.0 BLEU claims; evaluated independently using SacreBLEU 2.6.0 and chrF++ with beam search decoding ($k=4$).
6. **Live Attention Extraction**: Deep Learning Laboratory features real dynamic cross-attention heatmaps extracted directly during PyTorch decoding.

---

## Architectural Specifications

| Parameter | Configuration |
| :--- | :--- |
| **Model Type** | Seq2Seq Encoder-Decoder Transformer |
| **Reference Architecture** | Vaswani et al. (2017) *Attention Is All You Need* |
| **Total Trainable Parameters** | **6,033,408 (6.03 Million)** |
| **Encoder / Decoder Layers** | **3 Layers Encoder + 3 Layers Decoder** ($N = 6$ sub-stacks) |
| **Attention Heads** | **4 Heads** ($d_k = d_v = 64$) |
| **Hidden Dimension ($d_{\text{model}}$)** | **256** |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | **1024** |
| **Positional Encoding** | Sinusoidal continuous trigonometric frequencies |
| **Weight Tying** | Shared token embeddings and output linear generator matrix |
| **Languages Supported** | English (`en`), Hindi (`hi`), Tamil (`ta`), Telugu (`te`) |
| **Vocabulary Size** | 2,000 multilingual BPE subwords |
| **Hardware Execution** | Apple Silicon GPU (`mps`) / NVIDIA CUDA / CPU |

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
│   │   └── ModelInfoPage.tsx     # 6.03M Parameter Architecture Specifications
│   └── dist/                     # Compiled Production Assets
│
├── ml_lab/                       # Deep Learning Research Lab
│   ├── custom_transformer/       # Pure PyTorch Transformer Implementation
│   │   ├── attention.py          # ScaledDotProductAttention & MultiHeadAttention
│   │   ├── embeddings.py         # TokenEmbedding & Sinusoidal PositionalEncoding
│   │   ├── encoder.py            # EncoderLayer & TransformerEncoder Stack
│   │   ├── decoder.py            # DecoderLayer & TransformerDecoder Stack
│   │   └── transformer.py        # Seq2SeqTransformer with Greedy & Beam Search
│   ├── tokenizer/                # Multilingual BPE Tokenizer (From Scratch)
│   │   ├── language_registry.py  # Language Config & Control Tokens (<2en>, <2hi>, ...)
│   │   └── multilingual_tokenizer.py # BPE Subword Trainer & Serializer
│   ├── datasets/                 # Parallel Corpus & Preprocessing Pipeline
│   │   ├── data_pipeline.py      # Unicode Normalization, Filtering & Splits
│   │   └── curated_corpus.py     # 4-Way Parallel Corpus (en, hi, ta, te)
│   ├── training/                 # Model Training Pipeline
│   │   └── train_multilingual_transformer.py # 40-Epoch AdamW + Noam Scheduler Loop
│   ├── evaluation/               # Scientific Benchmark Suite
│   │   ├── evaluate.py           # SacreBLEU & chrF++ Benchmark Evaluator
│   │   └── results/              # Verified Benchmark Reports & Loss Histories
│   └── checkpoints/              # Trained PyTorch Model Checkpoints (.pt)
│
├── tests/                        # Comprehensive Unit & Architecture Test Suite
│   ├── test_custom_transformer.py # Masking, Attention & Shape Invariance Tests
│   ├── test_multilingual_tokenizer.py # Subword Encoding/Decoding Tests
│   ├── test_data_pipeline.py     # Preprocessing & Split Integrity Tests
│   └── test_api.py               # REST API Endpoint Tests
│
└── docs/                         # In-Depth Research Documentation
    ├── CURRENT_PROJECT_AUDIT.md  # Comprehensive Codebase Audit
    ├── CUSTOM_TRANSFORMER_MATHEMATICS.md # Equations, LayerNorm, Loss Formulations
    ├── TRAINING_EXPERIMENTS.md   # Loss Curves, Hyperparameters, Optimization
    ├── RESULTS.md                # SacreBLEU & chrF++ Evaluation on Held-Out Test Data
    ├── REPRODUCIBILITY.md        # Step-by-Step Reproduction Guide
    ├── architecture.md           # System Architecture & Component Interactions
    ├── model_documentation.md    # Parameter Math & Dimension Breakdown
    └── dataset_documentation.md  # Preprocessing Pipeline & Unicode Normalization
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
*(All 23 tests pass: architecture, causal masking, future token leakage, tokenizer, data pipeline, and API endpoints).*

### 3. Start Application Server
```bash
source .venv/bin/activate
python backend/app/main.py
```
Open **`http://localhost:8000`** to access the live web application.

---

## Empirical Benchmark Results (Held-Out Test Set)

| Language Direction | Source Script | Target Script | SacreBLEU | chrF++ | Avg Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **English → Hindi** (`en-hi`) | Latin | Devanagari | 0.40 | 3.31 | 1.46s |
| **English → Tamil** (`en-ta`) | Latin | Tamil | 0.86 | 11.17 | 1.43s |
| **English → Telugu** (`en-te`) | Latin | Telugu | 0.00 | 6.06 | 1.45s |
| **Hindi → English** (`hi-en`) | Devanagari | Latin | 0.00 | 5.48 | 1.48s |
| **Tamil → English** (`ta-en`) | Tamil | Latin | 0.00 | 4.54 | 1.55s |
| **Telugu → English** (`te-en`) | Telugu | Latin | 0.00 | 6.84 | 1.61s |

*Scores reflect evaluation on genuine independent held-out parallel test sentences using SacreBLEU 2.6.0 with beam search ($k=4$).*

---

## License

This project is licensed under the **MIT License**. The Transformer architecture is an independent academic implementation based on Vaswani et al. (2017).
