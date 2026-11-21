# Training Guide: PyTorch Custom Multilingual Transformer

This guide documents the full training pipeline, mathematical loss function, optimizer settings, and execution instructions for training the TranslateMe Seq2Seq Transformer from scratch.

---

## 1. Mathematical Objective & Loss Function

The model is trained via **Teacher Forcing** with shifted target labels. Target sequence $Y = (y_1, y_2, \dots, y_T)$ is fed into the decoder as:
$$Y_{\text{input}} = (\langle \text{sos} \rangle, y_1, y_2, \dots, y_{T-1})$$
And compared against target ground truth labels:
$$Y_{\text{target}} = (y_1, y_2, \dots, y_{T-1}, \langle \text{eos} \rangle)$$

### Label Smoothing Cross-Entropy
To prevent overconfident predictions and improve generalization across Indic subwords, label smoothing redistributes $\epsilon = 0.1$ probability uniformly:
$$q'(k) = (1 - \epsilon) \cdot \mathbb{I}(y = k) + \frac{\epsilon}{|V| - 1}$$
Padding tokens ($\text{pad\_id} = 0$) are ignored:
$$\mathcal{L} = -\frac{1}{N_{\text{valid}}} \sum_{t=1}^{T} \sum_{k \neq \text{pad}} q'(k) \log P(y_t = k \mid Y_{<t}, X)$$

---

## 2. Optimizer & Learning Rate Schedule

### AdamW Optimizer
- $\beta_1 = 0.9$
- $\beta_2 = 0.98$
- $\epsilon = 10^{-9}$
- $\text{weight\_decay} = 0.01$

### Noam Learning Rate Schedule
The learning rate follows the schedule from *Attention Is All You Need*:
$$\text{lr}(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$
- Warmup steps: $50$
- Gradient clipping: $\|\mathbf{g}\|_2 \le 1.0$

---

## 3. How to Execute Training

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run training script
python -m ml_lab.training.train_multilingual_transformer
```

### Outputs Generated:
- Checkpoint: `ml_lab/checkpoints/custom_multilingual_transformer.pt`
- Training History: `ml_lab/evaluation/results/custom_multilingual_training_history.json`

---

## 4. Hardware Acceleration

The training script automatically detects and utilizes:
1. **Apple Silicon GPU (`mps`)**: Metal Performance Shaders acceleration on macOS.
2. **NVIDIA GPU (`cuda`)**: CUDA execution if an NVIDIA GPU is available.
3. **CPU Fallback**: Vectorized multi-threaded CPU execution if no GPU is present.
