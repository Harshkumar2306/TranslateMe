# Dataset Provenance, Licensing & Attribution Guide

This document details the exact provenance, licensing terms, and attribution obligations for all parallel datasets utilized or evaluated in TranslateMe.

---

## 1. Licensing Matrix

| Dataset | Provider / Source | License | Permitted Use | Attribution Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **AI4Bharat Samanantar** | AI4Bharat / IIT Madras | **CC-BY 4.0** | Commercial & Non-Commercial with Attribution | Must cite Gowda et al. (2022) |
| **OPUS Parallel Corpora** | University of Helsinki (Tiedemann) | **CC-BY 4.0 / CC0 / GNU** | Research & Educational | Must cite Tiedemann (2012) |
| **FLORES-200 Benchmark** | Meta AI Research | **CC-BY-SA 4.0** | Evaluation Benchmark Only | Must cite NLLB Team et al. (2022) |
| **TranslateMe Curated Suite** | TranslateMe Academic DL Project | **MIT License** | Open Source / Unrestricted | Standard MIT Copyright Notice |

---

## 2. Dataset Citations & Attribution

### AI4Bharat Samanantar
> Gowda, T., Praddeep, N. G., et al. (2022).  
> **Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages.**  
> *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 6943–6960.  
> Official Repository: `https://github.com/AI4Bharat/Samanantar`

### OPUS Parallel Data Repository
> Tiedemann, J. (2012).  
> **Parallel Data, Tools and Interfaces in OPUS.**  
> *Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)*, pages 2214–2218.  
> Repository URL: `https://opus.nlpl.eu/`

### FLORES-200 Evaluation Suite
> NLLB Team, Marta R. Costa-jussà, et al. (2022).  
> **No Language Left Behind: Scaling Human-Centered Machine Translation.**  
> *Meta AI Research*, arXiv:2207.04672.  
> *Note:* FLORES-200 is referenced strictly as an evaluation benchmark standard, not for model training.

---

## 3. Strict Compliance Policy

1. **No Proprietary Crawls Without Attribution**: No scraping of copyrighted literary or commercial works.
2. **Evaluation-Only Isolation**: Benchmark sets (such as held-out test suites) must never be incorporated into tokenizer training, language modeling, or parameter optimization.
3. **Data Contamination Prevention**: All parallel sentence pairs are cryptographically checked against test sets prior to training.
