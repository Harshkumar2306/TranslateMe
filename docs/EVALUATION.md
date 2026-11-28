# Scientific Evaluation Methodology

This document describes the evaluation framework, metrics, and decoding settings used by `ml_lab/evaluation/evaluate.py` to benchmark the active TranslateMe **113.29M** Custom Multilingual Transformer.

---

## 1. Metric Formulations

### 1. SacreBLEU (Papineni et al., 2002; Post, 2018)
SacreBLEU computes modified $n$-gram precision with a brevity penalty to evaluate word-level translation fidelity:
$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$
where brevity penalty:
$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp\left(1 - \frac{r}{c}\right) & \text{if } c \le r \end{cases}$$

### 2. chrF++ (Popović, 2015, 2017)
chrF++ evaluates character $n$-grams (1 to 6-grams) combined with word unigrams/bigrams using F-score:
$$\text{chrF} = \left(1 + \beta^2\right) \frac{\text{chrP} \cdot \text{chrR}}{\beta^2 \text{chrP} + \text{chrR}}$$
For agglutinative Indic scripts (Devanagari, Tamil, Telugu), chrF++ captures morphological and stem agreement even when exact surface inflections differ from the single reference.

---

## 2. Decoding Settings

Evaluation matches production inference (`backend/app/services/custom_multilingual_service.py`):
- **Beam Width ($k$)**: `2`
- **Length Normalization**: $\text{LP}(L) = \left(\frac{5 + L}{6}\right)^{0.6}$
- **Repetition Penalty**: `1.15`
- **No-Repeat N-Gram Size**: `3`
- **Maximum Output Length**: `96` tokens
- **Script Masking**: Disallows foreign Indic scripts in target vocabulary logits
- **Target Conditioning**: Decoder prefix `[<bos>, <2lang>]` (`<2en>`, `<2hi>`, `<2ta>`, `<2te>`)
