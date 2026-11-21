# Reproducibility Guide: From Scratch to Evaluation

This guide provides step-by-step instructions to reproduce every stage of the TranslateMe Multilingual Neural Machine Translation system from complete random initialization to final evaluation.

---

## 1. System Requirements

- **Operating System**: macOS (Apple Silicon recommended for MPS acceleration) or Linux (CUDA-enabled GPU recommended) or CPU fallback.
- **Python**: 3.10 to 3.12
- **Node.js**: v18+ and npm (for frontend compilation)
- **Disk Space**: ~2 GB (virtualenv, dependencies, dataset, checkpoints)

---

## 2. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/Harshkumar2306/TranslateMe.git
cd TranslateMe

# 2. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install fastapi uvicorn tokenizers sentencepiece sacrebleu pytest httpx pydantic sqlalchemy jinja2 python-multipart
```

---

## 3. Train the Multilingual BPE Tokenizer from Scratch

Train a new 2,000-subword shared vocabulary over the 4-script corpus (`en`, `hi`, `ta`, `te`) with control tokens:

```bash
source .venv/bin/activate
python -m ml_lab.tokenizer.multilingual_tokenizer
```

**Verification:**
This generates `ml_lab/tokenizer/multilingual_tokenizer.json` and verifies encoding/decoding for Latin, Devanagari, Tamil, and Telugu scripts.

---

## 4. Train the Custom Transformer from Scratch

Train the 6.03M parameter Seq2Seq Transformer initialized from random Gaussian weights:

```bash
source .venv/bin/activate
python -m ml_lab.training.train_multilingual_transformer
```

**Process:**
- Trains for 40 epochs on `mps` (or `cuda` / `cpu`).
- Evaluates validation loss after each epoch.
- Applies Noam learning rate warmup and AdamW weight decay.
- Saves the best model checkpoint to `ml_lab/checkpoints/custom_multilingual_transformer.pt`.
- Saves full training history to `ml_lab/evaluation/results/custom_multilingual_training_history.json`.

---

## 5. Execute the Unit and Architecture Test Suite

Verify architecture components, tensor shapes, causal masking, future token leakage, tokenizer special tokens, and API endpoints:

```bash
source .venv/bin/activate
pytest tests/ -v
```

**Expected Outcome:**
23 passing tests:
- `tests/test_custom_transformer.py` (9 passed)
- `tests/test_multilingual_tokenizer.py` (2 passed)
- `tests/test_data_pipeline.py` (4 passed)
- `tests/test_api.py` (8 passed)

---

## 6. Run Scientific Evaluation on Held-Out Test Data

Compute unvarnished SacreBLEU and chrF++ benchmarks across all 6 language directions using beam search:

```bash
source .venv/bin/activate
python -m ml_lab.evaluation.evaluate
```

**Outputs:**
- Generates `ml_lab/evaluation/results/custom_multilingual_evaluation_report.json`.
- Generates comparative chart `ml_lab/evaluation/results/multilingual_bleu_chrf.png`.

---

## 7. Build and Run the Complete Application

### Compile Frontend Production Assets
```bash
npm --prefix frontend install
npm --prefix frontend run build
```

### Launch Unified FastApi + React Server
```bash
source .venv/bin/activate
python backend/app/main.py
```

Open your browser at `http://localhost:8000`:
- **Translator**: Test live translation across English, Hindi, Tamil, and Telugu.
- **DL Lab**: View live dynamic cross-attention heatmaps, empirical loss curves, and held-out test benchmarks.
- **Specs**: Inspect 6.03M parameter architecture specifications.
