# Model Documentation & Technical Specifications

## 1. NLLB-200 Multilingual Transformer

### Architecture
- **Model Family**: No Language Left Behind (NLLB)
- **Checkpoint**: `facebook/nllb-200-distilled-600M`
- **Architecture**: Sequence-to-Sequence Encoder-Decoder Transformer
- **Number of Parameters**: ~615 Million
- **Encoder Layers**: 12
- **Decoder Layers**: 12
- **Attention Heads**: 16
- **Hidden Dimension ($d_{model}$)**: 1024
- **Feed-Forward Dimension ($d_{ff}$)**: 4096
- **Tokenizer**: SentencePiece with 256,204 tokens

### Language Coverage
NLLB-200 supports 202 languages and distinct scripts, including all major Indian languages (Hindi `hin_Deva`, Tamil `tam_Taml`, Telugu `tel_Telu`, Malayalam `mal_Mlym`, Kannada `kan_Knda`, Bengali `ben_Beng`, Marathi `mar_Deva`, Gujarati `guj_Gujr`, Punjabi `pan_Guru`, Urdu `urd_Arab`, Odia `ory_Orya`, Assamese `asm_Beng`, Sanskrit `san_Deva`) and global languages (English, French, Spanish, German, Japanese, Korean, Chinese, Arabic, Russian).

### Research Attribution & License
- **Authors**: NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, et al.
- **Organization**: Meta AI Research (2022)
- **Publication**: *No Language Left Behind: Scaling Human-Centered Machine Translation*, arXiv:2207.04672.
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0).

---

## 2. Custom PyTorch Transformer

### Architecture
- **Reference**: Vaswani et al. (2017), *Attention Is All You Need*, NeurIPS 2017.
- **Module**: `models/custom_transformer/`
- **Components**:
  - `ScaledDotProductAttention`
  - `MultiHeadAttention`
  - `PositionalEncoding`
  - `TransformerEncoder`
  - `TransformerDecoder`
  - `Seq2SeqTransformer`
- **Optimization**:
  - Optimizer: Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$)
  - Learning Rate Schedule: Noam warmup scheduler ($lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$)
  - Loss Function: Cross-Entropy with padding token masking
