# Mathematical Foundations of the Custom Multilingual Transformer

This document provides a comprehensive mathematical formulation of the TranslateMe Sequence-to-Sequence Transformer, implemented independently in pure PyTorch (`ml_lab/custom_transformer/`) based on Vaswani et al. (2017), *Attention Is All You Need*.

---

## 1. Sequence-to-Sequence Modeling Overview

Neural Machine Translation (NMT) maps a source sentence $X = (x_1, x_2, \dots, x_{S})$ in source language $L_{\text{src}}$ to a target sentence $Y = (y_1, y_2, \dots, y_{T})$ in target language $L_{\text{tgt}}$.

Under an autoregressive factorization, the joint conditional probability is decomposed using the chain rule of probability:

$$
P(Y \mid X; \theta) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, X; \theta)
$$

where $y_{<t} = (y_1, \dots, y_{t-1})$, with $y_1 = \langle \text{sos} \rangle$ and $y_T = \langle \text{eos} \rangle$.

---

## 2. Input Representations and Embeddings

### 2.1 Token Embeddings with Square-Root Scaling
Given a sequence of discrete subword token indices $x = (x_1, \dots, x_S) \in \mathbb{N}^S$, each token is mapped to a continuous dense vector via an embedding lookup matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$:

$$
\mathbf{e}_i = E[x_i] \cdot \sqrt{d_{\text{model}}}
$$

Scaling by $\sqrt{d_{\text{model}}}$ prevents embedding values from being dominated by positional encodings when summed.

### 2.2 Sinusoidal Positional Encoding
Because self-attention contains no inherent recurrence or convolution, order information is injected using fixed sinusoidal position encodings $PE \in \mathbb{R}^{S \times d_{\text{model}}}$:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

where:
- $pos \in [0, S-1]$ is the token position in the sequence,
- $i \in [0, d_{\text{model}}/2 - 1]$ indexes the embedding dimension frequency band.

The property of trigonometric addition allows the model to learn relative position shifts:

$$
PE_{(pos + k, 2i)} = PE_{(pos, 2i)} \cos\left(k \cdot \omega_i\right) + PE_{(pos, 2i+1)} \sin\left(k \cdot \omega_i\right)
$$

The final input representation passed to the encoder or decoder stack is:

$$
Z^{(0)} = \text{Dropout}\left(\mathbf{E} + PE\right)
$$

---

## 3. Attention Mechanisms

### 3.1 Scaled Dot-Product Attention
Given queries $Q \in \mathbb{R}^{N \times d_k}$, keys $K \in \mathbb{R}^{M \times d_k}$, and values $V \in \mathbb{R}^{M \times d_v}$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

where $M \in \mathbb{R}^{N \times M}$ is an additive attention mask:
- $M_{ij} = 0$ for allowed attention connections,
- $M_{ij} = -\infty$ (or $-10^9$) for masked positions (padding or future lookahead).

Scaling by $\frac{1}{\sqrt{d_k}}$ counteracts the growth of dot products for large $d_k$, preventing the softmax from entering regions with extremely small gradients.

### 3.2 Multi-Head Attention
Instead of performing a single attention function with $d_{\text{model}}$-dimensional queries, keys, and values, Multi-Head Attention linearly projects queries, keys, and values $h$ times with parameter matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$:

$$
\text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}\left(\text{head}_1, \dots, \text{head}_h\right) W^O
$$

where $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$. In TranslateMe:
$$d_k = d_v = \frac{d_{\text{model}}}{h} = \frac{256}{4} = 64$$

---

## 4. Encoder and Decoder Layers

### 4.1 Sublayer Residual Connection and Layer Normalization
Each sublayer incorporates a residual connection followed by layer normalization (Post-LN formulation):

$$
\text{SublayerOutput}(x) = \text{LayerNorm}\left(x + \text{Dropout}\left(\text{Sublayer}(x)\right)\right)
$$

Layer Normalization computes mean $\mu$ and variance $\sigma^2$ across the feature dimension:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

where $\gamma, \beta \in \mathbb{R}^{d_{\text{model}}}$ are learnable affine parameters, and $\epsilon = 10^{-6}$.

### 4.2 Position-Wise Feed-Forward Networks
Each layer in the encoder and decoder contains a fully connected feed-forward network applied identically and independently to each sequence position:

$$
\text{FFN}(x) = \max\left(0, x W_1 + b_1\right) W_2 + b_2
$$

where $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $b_1 \in \mathbb{R}^{d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d_{\text{model}}}$. In TranslateMe, $d_{\text{ff}} = 1024$.

### 4.3 Decoder Causal Masking
To preserve the autoregressive property during training, self-attention in the decoder applies a causal mask $M_{\text{causal}} \in \mathbb{R}^{T \times T}$:

$$
M_{\text{causal}}[i, j] = 
\begin{cases} 
0 & \text{if } j \le i \\ 
-\infty & \text{if } j > i 
\end{cases}
$$

This prevents position $i$ from attending to subsequent tokens $j > i$.

### 4.4 Decoder Cross-Attention
In decoder cross-attention, queries are derived from the previous decoder layer representations $Z_{\text{dec}}^{(l-1)}$, while keys and values are derived from the encoder output memory $Z_{\text{enc}}^{(N_{\text{enc}})}$:

$$
Q = Z_{\text{dec}}^{(l-1)} W_i^Q, \quad K = Z_{\text{enc}}^{(N_{\text{enc}})} W_i^K, \quad V = Z_{\text{enc}}^{(N_{\text{enc}})} W_i^V
$$

This enables every target token position to attend dynamically to all source sequence tokens.

---

## 5. Weight Tying and Linear Generator

The output probabilities are computed via a linear projection followed by a log-softmax:

$$
P(y_t \mid y_{<t}, X) = \text{softmax}\left(Z_{\text{dec}, t}^{(N_{\text{dec}})} W_{\text{proj}} + b_{\text{proj}}\right)
$$

In TranslateMe, the input token embedding matrix $E$ and the output projection matrix $W_{\text{proj}}$ share identical weights ($W_{\text{proj}} = E^T$), reducing total model parameter count and improving generalization:

$$
\text{Params}_{\text{tied}} = |V| \cdot d_{\text{model}} \quad \text{instead of} \quad 2 \cdot |V| \cdot d_{\text{model}}
$$

---

## 6. Training Loss and Optimization

### 6.1 Label Smoothing Cross-Entropy
Given ground-truth one-hot distribution $q(k)$, label smoothing redistributes $\epsilon_{\text{ls}} = 0.1$ probability uniformly across all $|V|$ vocabulary classes:

$$
q'(k) = (1 - \epsilon_{\text{ls}}) q(k) + \frac{\epsilon_{\text{ls}}}{|V| - 1}
$$

The padding index $\text{pad\_id} = 0$ is excluded from loss computation:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N_{\text{tokens}}} \sum_{t=1}^{T} \sum_{k \neq \text{pad}} q'(k) \log P(y_{t} = k \mid y_{<t}, X)
$$

### 6.2 Noam Learning Rate Schedule
The AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}, \text{weight\_decay} = 0.01$) updates parameters according to the Noam learning rate schedule:

$$
\text{lr}(\text{step}) = d_{\text{model}}^{-0.5} \cdot \min\left(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)
$$

For step $\le \text{warmup\_steps}$, learning rate increases linearly. For step $> \text{warmup\_steps}$, it decays proportionally to $\frac{1}{\sqrt{\text{step}}}$.

---

## 7. Inference and Decoding Algorithms

### 7.1 Greedy Decoding
At each step $t$, greedy decoding selects the token with the highest predicted log-probability:

$$
\hat{y}_t = \arg\max_{k \in V} P(y_t = k \mid \hat{y}_{<t}, X)
$$

To prevent cyclic loops in small-corpus regimes, a repetition penalty $\gamma = 1.2$ modifies logits for previously generated tokens $k \in \{\hat{y}_{1}, \dots, \hat{y}_{t-1}\}$:

$$
\text{logit}'(k) = 
\begin{cases} 
\text{logit}(k) / \gamma & \text{if } \text{logit}(k) > 0 \\ 
\text{logit}(k) \cdot \gamma & \text{if } \text{logit}(k) \le 0 
\end{cases}
$$

### 7.2 Beam Search with Length Normalization
Beam search maintains a set of $B$ top hypothesis sequences $\mathcal{H}_t = \{Y^{(1)}, \dots, Y^{(B)}\}$. At step $t$, candidates are ranked by length-normalized cumulative log-probability:

$$
\text{Score}(Y) = \frac{\sum_{i=1}^{|Y|} \log P(y_i \mid y_{<i}, X)}{\text{LP}(|Y|)}
$$

where the length penalty $\text{LP}$ prevents bias toward shorter sequences:

$$
\text{LP}(L) = \left(\frac{5 + L}{6}\right)^\alpha
$$

TranslateMe uses $\alpha = 0.6$, beam width $B \in [2, 4]$, and early stopping when $B$ completed hypotheses end with $\langle \text{eos} \rangle$.
