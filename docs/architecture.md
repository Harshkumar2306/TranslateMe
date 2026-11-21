# TranslateMe: System Architecture & Design

## 1. High-Level Architectural Overview

TranslateMe is structured around decoupled components that maintain strict separation of concerns across Deep Learning models, REST API microservices, persistent relational storage, and an interactive React frontend:

```
+-----------------------------------------------------------------------+
|                           React Frontend                              |
|           (TypeScript + Vite + Tailwind CSS + Recharts)               |
+-----------------------------------+-----------------------------------+
                                    | HTTP / REST API (JSON)
                                    v
+-----------------------------------------------------------------------+
|                        FastAPI Backend Engine                         |
|  - Request Validation (Pydantic)   - CORS Middleware                  |
|  - Language Detection Service      - Document Processing (PyMuPDF)    |
|  - Rate & Length Boundary Checks   - Lifecycle Model Warmup           |
+------------------+--------------------------------+-------------------+
                   |                                |
                   v                                v
+-----------------------------------+   +-------------------------------+
|     Translation Engine (PyTorch)   |   |   SQLite History Database     |
| - NLLB-200 (facebook/600M)        |   |   (SQLAlchemy ORM)            |
|   * 202 Languages Direct Mapping  |   |   * Source & Translated Text  |
|   * Beam Search & Alternatives    |   |   * Latency Metrics           |
| - Custom Seq2Seq Transformer      |   |   * Language Distributions    |
|   * Sinusoidal Positional Encoding|   +-------------------------------+
|   * Scaled Dot-Product Attention  |
|   * Multi-Head Attention Heads    |
+-----------------------------------+
```

---

## 2. Deep Learning Subsystems

### A. Pretrained Multilingual NLLB-200 Transformer
- **Base Model**: `facebook/nllb-200-distilled-600M`
- **Parameters**: 615,073,792 weights
- **Vocabulary**: 256,204 tokens with SentencePiece Byte-Pair Encoding
- **Direct Many-to-Many Translation**: Supports arbitrary pairs (e.g. Hindi $\rightarrow$ Tamil, Tamil $\rightarrow$ French, Telugu $\rightarrow$ Japanese) without translating through English as an intermediary pivot language.
- **Forced Target Generation**: Employs target language special tokens passed as `forced_bos_token_id` to guide autoregressive decoding.

### B. Custom Transformer (Vaswani et al., 2017)
- Located in `models/custom_transformer/`
- Implements:
  1. `TokenEmbedding`: Input embedding scaled by $\sqrt{d_{model}}$.
  2. `PositionalEncoding`: Sinusoidal trigonometric wave positions.
  3. `ScaledDotProductAttention`: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$.
  4. `MultiHeadAttention`: Parallel projections across $h$ subspaces.
  5. `PositionwiseFeedForward`: Two-layer linear feed-forward network with ReLU/GELU.
  6. `SublayerConnection`: Residual additions with Layer Normalization.
  7. `Seq2SeqTransformer`: Encoder-Decoder integration with causal triangular masking.

---

## 3. Storage and Database Schema
- **Database**: SQLite 3 (`lingua_ai.db`)
- **ORM**: SQLAlchemy
- **Table**: `translation_history`
  - `id`: Integer primary key (autoincrement)
  - `source_text`: Raw input text
  - `translated_text`: Autoregressively generated translation
  - `source_language`: BCP-47 / Flores language code
  - `source_language_name`: Human-readable source language
  - `target_language`: Target language code
  - `target_language_name`: Human-readable target language
  - `detected_language`: Automatically identified language code if auto-detect was used
  - `inference_time`: High precision inference latency (seconds)
  - `model_name`: Model identifier
  - `created_at`: UTC timestamp
