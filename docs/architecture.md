# TranslateMe: System Architecture & Design

## 1. High-Level Architectural Overview

TranslateMe is a genuinely from-scratch multilingual Neural Machine Translation (NMT) system built on an independent PyTorch Transformer implementation, a custom Byte-Pair Encoding (BPE) subword tokenizer, an asynchronous FastAPI backend, and an interactive React + Tailwind CSS research laboratory.

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
|  - Language Routing & Control      - Live Attention Extraction API    |
|  - Rate & Length Boundary Checks   - Lifecycle Custom Model Loader    |
+------------------+--------------------------------+-------------------+
                   |                                |
                   v                                v
+-----------------------------------+   +-------------------------------+
|  Custom Seq2Seq Transformer       |   |   SQLite History Database     |
|  (Pure PyTorch Modules)           |   |   (SQLAlchemy ORM)            |
|  - 6,033,408 Trainable Weights    |   |   * Source & Translated Text  |
|  - 3 Encoder + 3 Decoder Layers   |   |   * Per-Inference Latency     |
|  - 4 Multi-Head Attention Heads   |   |   * Language Distributions    |
|  - Sinusoidal Positional Encoding |   +-------------------------------+
|  - Shared Tied Embeddings (2,000) |
|  - Autoregressive Beam Search     |
+-----------------------------------+
```

---

## 2. Deep Learning Subsystem (`ml_lab/custom_transformer/`)

The core translation engine is implemented independently in pure PyTorch without relying on `torch.nn.Transformer` or Hugging Face pretrained models:

1. **`ScaledDotProductAttention` (`attention.py`)**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
   Supports dynamic cross-attention matrix extraction (`return_attn=True`).

2. **`MultiHeadAttention` (`attention.py`)**:
   Splits queries, keys, and values into $h = 4$ representation subspaces with $d_k = d_v = 64$, concatenated and projected through $W^O$.

3. **`TokenEmbedding` & `PositionalEncoding` (`embeddings.py`)**:
   Scales subword embeddings by $\sqrt{d_{\text{model}}} = \sqrt{256} = 16$. Adds continuous sinusoidal frequencies across positions.

4. **`TransformerEncoder` & `EncoderLayer` (`encoder.py`)**:
   $N = 3$ identical encoder blocks with multi-head self-attention, position-wise feed-forward ($d_{\text{ff}} = 1024$), residual connections, and layer normalization.

5. **`TransformerDecoder` & `DecoderLayer` (`decoder.py`)**:
   $N = 3$ identical decoder blocks with masked self-attention (causal lower-triangular mask), cross-attention over encoder memory, and feed-forward networks.

6. **`Seq2SeqTransformer` (`transformer.py`)**:
   Ties input subword embeddings with the output linear generator, implements autoregressive greedy decoding with repetition penalty, and beam search decoding with length penalty normalization.

---

## 3. Multilingual Tokenization (`ml_lab/tokenizer/`)

- **Subword BPE Model**: Trained from scratch over parallel corpus using Hugging Face `tokenizers`.
- **Vocabulary Size**: 2,000 subwords supporting Latin (English), Devanagari (Hindi), Tamil, and Telugu scripts.
- **Language Direction Control Tokens**: `<2en>`, `<2hi>`, `<2ta>`, `<2te>`.
- **Special Tokens**: `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).

---

## 4. Storage and Database Schema
- **Database**: SQLite 3 (`lingua_ai.db`)
- **ORM**: SQLAlchemy
- **Table**: `translation_history`
  - `id`: Primary key autoincrement
  - `source_text`: Input text
  - `translated_text`: Autoregressively generated translation
  - `source_language`: Language code (`en`, `hi`, `ta`, `te`)
  - `source_language_name`: English name
  - `target_language`: Target language code
  - `target_language_name`: Target English name
  - `inference_time`: Latency in seconds
  - `model_name`: Model identifier
  - `created_at`: Timestamp
