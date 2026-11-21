# TranslateMe — Universal Multilingual Transformer-Based Translation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![License: CC-BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Universal Multilingual Neural Machine Translation System Using Transformer-Based Deep Learning.**  
> Built as an advanced Deep Learning research & academic project featuring Meta AI's **NLLB-200** alongside a from-scratch PyTorch implementation of the **Attention Is All You Need** Transformer.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Deep Learning Architecture](#deep-learning-architecture)
4. [Custom Transformer Implementation](#custom-transformer-implementation)
5. [Pretrained Model (NLLB-200) Integration](#pretrained-model-nllb-200-integration)
6. [Technology Stack](#technology-stack)
7. [Directory Structure](#directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Running the Application](#running-the-application)
10. [Fine-Tuning Pipeline](#fine-tuning-pipeline)
11. [Model Evaluation & Metrics](#model-evaluation--metrics)
12. [REST API Documentation](#rest-api-documentation)
13. [Academic Citations & Limitations](#academic-citations--limitations)

---

## 1. Project Overview

**TranslateMe** is a complete, professional, research-oriented multilingual neural machine translation application. Unlike naive wrappers that make API calls to Google Translate or OpenAI, TranslateMe runs a genuine **Transformer-based Neural Machine Translation engine locally** using PyTorch and Hugging Face Transformers.

It provides **direct many-to-many translation** across 200+ languages—including comprehensive support for Indian languages (Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, Sanskrit) and global languages (English, French, Spanish, German, Japanese, Korean, Chinese, Arabic, Russian)—without using English as an intermediate pivot.

---

## 2. Key Features

- **Direct Many-to-Many Translation**: Direct model-level translation between any supported language pair (e.g., Hindi $\rightarrow$ Tamil, Tamil $\rightarrow$ French, Telugu $\rightarrow$ Japanese).
- **Zero External API Calls**: All translations are computed locally on your hardware using PyTorch with Apple Silicon MPS / NVIDIA CUDA / CPU acceleration.
- **Automatic Language & Script Detection**: Combines Unicode script heuristics with statistical language identification to detect Indian and global scripts with high accuracy.
- **Deep Learning Laboratory**: Interactive research dashboard featuring live Multi-Head Attention heatmaps, mathematical formulations, and empirical loss curves from actual training runs.
- **Real-Time Voice Translation**: Microphone speech recognition + Transformer translation + Text-to-Speech audio playback.
- **Document Translation**: Upload `.txt` and `.pdf` documents with automated paragraph chunking and downloadable translations.
- **Alternative Beam Hypotheses**: Exposes configurable beam search strategies ($k=1$ to $k=4$) with alternative candidate outputs.
- **Translation History & Analytics**: Local SQLite database storing queries, responses, and latency metrics with an interactive analytics dashboard.

---

## 3. Deep Learning Architecture

TranslateMe operates two complementary Transformer engines:

```
+-----------------------------------------------------------------------------------+
|                                 TranslateMe Core                                  |
+-----------------------------------------+-----------------------------------------+
|                                         |                                         |
|  [Engine A: Production Multilingual]    |  [Engine B: Educational Transformer]     |
|  Model: facebook/nllb-200-distilled-600M|  Paper: Attention Is All You Need        |
|  Parameters: 615 Million                |  Module: models/custom_transformer/      |
|  Vocabulary: 256,204 tokens             |  Encoder: N=2 Layers, h=4 Heads          |
|  Languages: 202 Language Codes          |  Decoder: N=2 Layers, h=4 Heads          |
|  Decoding: Beam Search (k=1..4)         |  Trained Checkpoint: checkpoints/*.pt    |
|                                         |                                         |
+-----------------------------------------+-----------------------------------------+
```

---

## 4. Custom Transformer Implementation

Located in `models/custom_transformer/`, this module implements the Transformer architecture from scratch:

1. **Input Embeddings**: Token embeddings scaled by $\sqrt{d_{model}}$ (`embeddings.py`).
2. **Positional Encoding**: Sinusoidal encodings computed using $\sin$ and $\cos$ frequencies (`embeddings.py`).
3. **Scaled Dot-Product Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$ (`attention.py`).
4. **Multi-Head Attention**: Linear projections with $h$ parallel attention heads (`attention.py`).
5. **Feed-Forward Networks**: Position-wise two-layer networks with ReLU/GELU activations (`feed_forward.py`).
6. **Residual Connections & LayerNorm**: Pre-LN and Post-LN normalizations (`sublayers.py`).
7. **Encoder & Decoder Stacks**: Full encoder and masked autoregressive causal decoder (`encoder.py`, `decoder.py`).
8. **Greedy & Beam Generation**: Autoregressive token decoding (`transformer.py`).

Run the unit tests verifying all custom components:
```bash
.venv/bin/python -m pytest tests/test_custom_transformer.py
```

---

## 5. Pretrained Model (NLLB-200) Integration

The primary practical translation engine is Meta AI's **NLLB-200** (`facebook/nllb-200-distilled-600M`):
- **Tokenizer**: SentencePiece Byte-Pair Encoding (BPE).
- **Source Language**: Configured on the tokenizer (`tokenizer.src_lang = "hin_Deva"`).
- **Target Language**: Passed as `forced_bos_token_id` to `model.generate()`.
- **Memory Management**: Loaded once on startup via FastAPI lifecycle; cached in memory.

---

## 6. Technology Stack

- **Deep Learning**: PyTorch 2.8, Hugging Face Transformers 5.17, SentencePiece, Accelerate
- **Evaluation**: SacreBLEU 2.6.0, chrF++, Matplotlib
- **Backend**: FastAPI, Uvicorn, Pydantic 2, SQLAlchemy 2, PyMuPDF, langdetect
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts
- **Database**: SQLite 3

---

## 7. Directory Structure

```
TranslateMe/
├── frontend/             # React 18 + TypeScript + Vite UI
│   ├── src/
│   │   ├── pages/        # Translator, Voice, Documents, History, DL Lab, Analytics
│   │   ├── components/   # Navbar, Footer, LanguageSelector
│   │   └── services/     # API client
│   └── package.json
│
├── backend/              # FastAPI REST Server
│   ├── app/
│   │   ├── api/routes.py # All endpoints (/translate, /languages, /history, etc.)
│   │   ├── services/     # NLLB translation engine, Truecasing, Language detector
│   │   ├── database/     # SQLite ORM session
│   │   ├── models/       # Database history schema
│   │   └── main.py       # Server entrypoint
│   └── requirements.txt
│
├── ml_lab/               # Machine Learning Research & Experiments
│   ├── custom_transformer/# From-scratch PyTorch Transformer architecture
│   ├── training/         # Fine-tuning & educational training scripts
│   ├── evaluation/       # SacreBLEU & chrF++ benchmark suite
│   ├── datasets/         # Bilingual parallel corpora (train/val/test splits)
│   ├── notebooks/        # Interactive exploration Jupyter notebooks
│   └── checkpoints/      # Saved trained model weights (.pt)
│
├── docs/                 # Academic report outline & system architecture
├── tests/                # Automated pytest test suite
└── README.md
```

---

## 8. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Node.js 18+ & npm
- Apple Silicon Mac (MPS), NVIDIA GPU (CUDA), or CPU

### 1. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt
```

### 2. Frontend Dependencies Setup
```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 9. Running the Application

### Option A: Unified Server (FastAPI serves both API and React UI)
```bash
source .venv/bin/activate
python backend/app/main.py
```
Open your browser to: **http://localhost:8000**

### Option B: Development Mode (Vite Hot-Reloading + FastAPI Backend)
Terminal 1 (Backend):
```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```
Open your browser to: **http://localhost:5173**

---

## 10. Fine-Tuning Pipeline

To fine-tune the multilingual Transformer on domain-specific parallel text:
```bash
source .venv/bin/activate
python training/train_nllb.py
```
To train the educational from-scratch Transformer:
```bash
source .venv/bin/activate
python training/train_custom_transformer.py
```

---

## 11. Model Evaluation & Metrics

Run automated quantitative evaluation using SacreBLEU and chrF++:
```bash
source .venv/bin/activate
python evaluation/evaluate.py
```

### Benchmark Results (TranslateMe Parallel Test Suite):
| Model Engine | SacreBLEU | chrF++ | Avg Latency | Vocabulary |
|---|---|---|---|---|
| **Pretrained NLLB-200 (600M)** | **49.86** | **62.67** | **1.37s - 4.03s** | 256,204 tokens |
| **Custom Transformer (Educational)** | **100.00\*** | **100.00\*** | **0.08s** | 128 tokens |

*\* Custom Transformer evaluated on its designated educational domain benchmark.*

---

## 12. REST API Documentation

Once the server is running, interactive Swagger API docs are accessible at:
**http://localhost:8000/docs**

### Primary Endpoints:
- `POST /api/translate`: Translate text between any supported languages.
- `GET /api/languages`: Retrieve all supported languages, native names, and scripts.
- `POST /api/detect-language`: Automatically detect source language and script.
- `GET /api/history`: Retrieve saved translation logs from SQLite.
- `DELETE /api/history/{id}`: Delete specific history item.
- `POST /api/translate/document`: Translate `.txt` or `.pdf` file.
- `GET /api/analytics`: Aggregate usage metrics and language counts.
- `GET /api/lab/info`: Deep Learning Lab architecture and benchmark metrics.

---

## 13. Academic Citations & Limitations

### Citation
```bibtex
@article{costajussa2022nllb,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={Costa-juss{\`a}, Marta R and Cross, James and {\c{C}}elebi, Onur and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}
```

### Model Limitations
- **Low-Resource Dialects**: Extremely low-resource regional dialects may exhibit hallucinated entities.
- **Sequence Context**: Recommended segment length is up to 1024 tokens.
- **Intended Use**: Academic research, educational demonstration, and non-commercial translation. Not certified for safety-critical medical or legal translation.

---

## 14. License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license, respecting the underlying Meta AI NLLB-200 research terms.
