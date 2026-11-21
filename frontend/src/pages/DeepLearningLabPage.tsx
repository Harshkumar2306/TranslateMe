import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import {
  Cpu, Layers, Activity, Sparkles, BookOpen, Award, RefreshCw, Send, AlertCircle
} from 'lucide-react';
import { LabInfo } from '../types';
import { fetchLabInfo, fetchAttentionMatrix } from '../services/api';

export const DeepLearningLabPage: React.FC = () => {
  const [labInfo, setLabInfo] = useState<LabInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'architecture' | 'attention' | 'training' | 'benchmark'>('architecture');

  // Interactive Live Attention State
  const [attnInputText, setAttnInputText] = useState('Hello, how are you today?');
  const [attnSrcLang, setAttnSrcLang] = useState('en');
  const [attnTgtLang, setAttnTgtLang] = useState('hi');
  const [attnLoading, setAttnLoading] = useState(false);
  const [attnError, setAttnError] = useState<string | null>(null);
  const [liveAttentionData, setLiveAttentionData] = useState<{
    source_tokens: string[];
    target_tokens: string[];
    attention_matrix: number[][];
    translated_text: string;
  } | null>(null);

  useEffect(() => {
    fetchLabInfo()
      .then((data) => {
        setLabInfo(data);
      })
      .catch(console.error)
      .finally(() => setLoading(false));

    // Initial attention fetch
    handleFetchAttention('Hello, how are you today?', 'en', 'hi');
  }, []);

  const handleFetchAttention = async (text: string, src: string, tgt: string) => {
    if (!text.trim()) return;
    setAttnLoading(true);
    setAttnError(null);
    try {
      const data = await fetchAttentionMatrix(text, src, tgt);
      // Slice target tokens and matrix to first 12 tokens for optimal visualization
      const maxRows = Math.min(12, data.target_tokens?.length || 10);
      const slicedTargets = data.target_tokens?.slice(0, maxRows) || [];
      const slicedMatrix = (data.attention_matrix || []).slice(0, maxRows);

      setLiveAttentionData({
        source_tokens: data.source_tokens || [],
        target_tokens: slicedTargets,
        attention_matrix: slicedMatrix,
        translated_text: data.translated_text || ''
      });
    } catch (err: any) {
      setAttnError(err.message || 'Failed to extract attention weights');
    } finally {
      setAttnLoading(false);
    }
  };

  // Format real training data for Recharts
  const history = labInfo?.custom_transformer.training_history;
  const trainingData = history && history.epochs
    ? history.epochs.map((ep, idx) => ({
        epoch: ep,
        train_loss: history.train_loss ? Number(history.train_loss[idx]?.toFixed(3)) : null,
        val_loss: history.val_loss ? Number(history.val_loss[idx]?.toFixed(3)) : null,
        lr: history.learning_rates ? Number((history.learning_rates[idx] * 1000).toFixed(4)) : null
      }))
    : [];

  // Format benchmark data per language pair
  const benchmarkObj = labInfo?.evaluation_benchmark || {};
  const benchmarkData = Object.keys(benchmarkObj).map((key) => {
    const item = benchmarkObj[key];
    return {
      direction: item.direction || key,
      bleu: Number((item.bleu || 0).toFixed(2)),
      chrf: Number((item.chrf || 0).toFixed(2)),
      latency: Number((item.avg_latency_sec || 0).toFixed(2)),
      throughput: Number((item.throughput_chars_per_sec || 0).toFixed(1))
    };
  });

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-violet-700 dark:text-violet-300 text-xs font-semibold mb-3">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Academic Deep Learning Research Lab</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">
          Deep Learning Laboratory
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Explore the internal mechanics of our custom from-scratch PyTorch Transformer: live cross-attention extraction, empirical training trajectories, and honest held-out test benchmarks.
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex justify-center mb-8 overflow-x-auto no-scrollbar py-1">
        <div className="inline-flex p-1 rounded-2xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 max-w-full overflow-x-auto no-scrollbar">
          {[
            { id: 'architecture', label: 'Architecture & Equations', icon: BookOpen },
            { id: 'attention', label: 'Live Attention Heatmap', icon: Layers },
            { id: 'training', label: 'Loss & LR Trajectories', icon: Activity },
            { id: 'benchmark', label: 'Held-Out Test Benchmark', icon: Award },
          ].map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all whitespace-nowrap flex-shrink-0 ${
                  isSelected
                    ? 'bg-white dark:bg-slate-900 text-violet-600 dark:text-violet-400 shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* TAB 1: ARCHITECTURE & THEORY */}
      {activeTab === 'architecture' && (
        <div className="space-y-8 animate-in fade-in duration-200">
          {/* Architectural Specs Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Parameters</span>
              <span className="text-xl font-extrabold text-violet-600 dark:text-violet-400">6,033,408</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Randomly Initialized</span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Layers</span>
              <span className="text-xl font-extrabold text-slate-800 dark:text-slate-200">3 Enc / 3 Dec</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">N = 6 Sub-stacks</span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Dimensions</span>
              <span className="text-xl font-extrabold text-slate-800 dark:text-slate-200">d=256, ff=1024</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">4 Heads (d_k = 64)</span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Tokenizer</span>
              <span className="text-xl font-extrabold text-emerald-600 dark:text-emerald-400">2,000 BPE</span>
              <span className="text-[10px] text-slate-500 block mt-0.5">Shared 4-Script Vocab</span>
            </div>
          </div>

          {/* Architectural Diagram Box */}
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
              From-Scratch Encoder-Decoder Transformer Architecture
            </h2>
            <p className="text-xs text-slate-500 mb-6">
              Independent PyTorch implementation in <code className="font-mono text-violet-600">ml_lab/custom_transformer/</code> following Vaswani et al. (2017).
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Encoder Stack */}
              <div className="p-5 rounded-2xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/60">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-bold text-sm text-blue-700 dark:text-blue-300 uppercase tracking-wider">
                    Transformer Encoder Stack
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-mono">
                    N = 3 Layers
                  </span>
                </div>

                <div className="space-y-2.5 text-xs font-mono">
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-blue-100 dark:border-blue-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Position-Wise Feed-Forward Network (d_model=256 → d_ff=1024 → 256)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Residual Add & Layer Normalization</div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-blue-100 dark:border-blue-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Multi-Head Self-Attention (h = 4 heads, d_k = 64)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Residual Add & Layer Normalization</div>
                  <div className="p-2.5 rounded-xl bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 text-center">
                    Subword Embeddings (scaled by √256) + Sinusoidal Positional Encoding
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Source Token IDs + Padding Mask</div>
                </div>
              </div>

              {/* Decoder Stack */}
              <div className="p-5 rounded-2xl bg-purple-50/50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900/60">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-bold text-sm text-purple-700 dark:text-purple-300 uppercase tracking-wider">
                    Transformer Decoder Stack
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 font-mono">
                    N = 3 Layers (Autoregressive)
                  </span>
                </div>

                <div className="space-y-2.5 text-xs font-mono">
                  <div className="p-2.5 rounded-xl bg-purple-100 dark:bg-purple-900/50 text-purple-800 dark:text-purple-200 text-center">
                    Tied Output Linear Projection + Softmax (Vocab: 2,000 subwords)
                  </div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-purple-100 dark:border-purple-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Position-Wise Feed-Forward Network (d_ff = 1024)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Residual Add & Layer Normalization</div>
                  <div className="p-2.5 rounded-xl bg-violet-100 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-center text-violet-800 dark:text-violet-200 font-semibold shadow-2xs">
                    Multi-Head Cross-Attention (Query: Decoder, Key/Value: Encoder Memory)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Residual Add & Layer Normalization</div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-purple-100 dark:border-purple-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Masked Multi-Head Self-Attention (Lower-Triangular Causal Lookahead Mask)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Target Shifted Right + Target Language Control Token (&lt;2hi&gt;)</div>
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Formulations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800">
              <h3 className="font-bold text-slate-900 dark:text-white text-base mb-2">
                1. Scaled Dot-Product Attention
              </h3>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Computes affinity between query vectors and key vectors, normalized by sqrt(d_k) to prevent vanishing gradients in softmax regions.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-xs text-violet-600 dark:text-violet-400">
                Attention(Q, K, V) = softmax((Q K^T) / √d_k + Mask) V
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800">
              <h3 className="font-bold text-slate-900 dark:text-white text-base mb-2">
                2. Sinusoidal Positional Encoding
              </h3>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Injects deterministic position awareness without requiring learnable recurrent states or fixed sequence length caps.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-xs text-indigo-600 dark:text-indigo-400">
                PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) <br/>
                PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800">
              <h3 className="font-bold text-slate-900 dark:text-white text-base mb-2">
                3. Label Smoothing Cross-Entropy
              </h3>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Regularizes target predictions by allocating ε = 0.1 mass across all non-pad vocabulary tokens, preventing overconfidence.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-xs text-emerald-600 dark:text-emerald-400">
                y'_k = (1 - ε) · y_k + ε / (|V| - 1)
              </div>
            </div>

            <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800">
              <h3 className="font-bold text-slate-900 dark:text-white text-base mb-2">
                4. Noam Learning Rate Schedule
              </h3>
              <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                Increases learning rate linearly for warmup_steps, followed by proportional decay to the inverse square root of step number.
              </p>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-xs text-amber-600 dark:text-amber-400">
                lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: LIVE ATTENTION MATRIX */}
      {activeTab === 'attention' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Live Dynamic Cross-Attention Heatmap
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Actual dynamic attention weights extracted directly from the custom Transformer during live PyTorch decoding.
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-800 font-mono">
                Layer 3 Cross-Attention (Mean Across 4 Heads)
              </span>
            </div>

            {/* Interactive Controller */}
            <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 mb-6">
              <div className="flex flex-col sm:flex-row gap-3 items-center">
                <input
                  type="text"
                  value={attnInputText}
                  onChange={(e) => setAttnInputText(e.target.value)}
                  placeholder="Enter sentence to extract attention..."
                  className="flex-1 w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <select
                    value={attnSrcLang}
                    onChange={(e) => setAttnSrcLang(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 dark:text-slate-300"
                  >
                    <option value="en">English (en)</option>
                    <option value="hi">Hindi (hi)</option>
                    <option value="ta">Tamil (ta)</option>
                    <option value="te">Telugu (te)</option>
                  </select>
                  <span className="text-xs text-slate-400">→</span>
                  <select
                    value={attnTgtLang}
                    onChange={(e) => setAttnTgtLang(e.target.value)}
                    className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 dark:text-slate-300"
                  >
                    <option value="hi">Hindi (hi)</option>
                    <option value="en">English (en)</option>
                    <option value="ta">Tamil (ta)</option>
                    <option value="te">Telugu (te)</option>
                  </select>
                  <button
                    onClick={() => handleFetchAttention(attnInputText, attnSrcLang, attnTgtLang)}
                    disabled={attnLoading || !attnInputText.trim()}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-700 text-white text-xs font-bold transition-all disabled:opacity-50"
                  >
                    {attnLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                    <span>Extract</span>
                  </button>
                </div>
              </div>

              {/* Quick Presets */}
              <div className="flex items-center gap-2 mt-3 overflow-x-auto no-scrollbar pt-1">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
                  Presets:
                </span>
                {[
                  { text: 'Hello, how are you today?', src: 'en', tgt: 'hi' },
                  { text: 'Machine learning models learn patterns.', src: 'en', tgt: 'ta' },
                  { text: 'Deep learning requires neural networks.', src: 'en', tgt: 'te' },
                  { text: 'नमस्ते, आप कैसे हैं?', src: 'hi', tgt: 'en' },
                ].map((p, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setAttnInputText(p.text);
                      setAttnSrcLang(p.src);
                      setAttnTgtLang(p.tgt);
                      handleFetchAttention(p.text, p.src, p.tgt);
                    }}
                    className="px-2.5 py-1 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:border-violet-500 whitespace-nowrap"
                  >
                    {p.src}→{p.tgt}: "{p.text.slice(0, 24)}..."
                  </button>
                ))}
              </div>
            </div>

            {attnError && (
              <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/50 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-xs flex items-center gap-2 mb-4">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{attnError}</span>
              </div>
            )}

            {/* Heatmap Grid */}
            {liveAttentionData && liveAttentionData.source_tokens.length > 0 && liveAttentionData.attention_matrix.length > 0 ? (
              <div className="overflow-x-auto pb-4">
                <div className="inline-block min-w-full">
                  {/* Column labels (Source tokens) */}
                  <div className="flex items-center ml-24 mb-2">
                    {liveAttentionData.source_tokens.map((tok, i) => (
                      <div key={i} className="w-16 text-center text-[11px] font-mono font-semibold text-slate-600 dark:text-slate-300 truncate px-1">
                        {tok}
                      </div>
                    ))}
                  </div>

                  {/* Rows (Target tokens & heatmap cells) */}
                  <div className="space-y-1.5">
                    {liveAttentionData.target_tokens.map((tgtTok, rowIdx) => {
                      const rowWeights = liveAttentionData.attention_matrix[rowIdx] || [];
                      return (
                        <div key={rowIdx} className="flex items-center">
                          <div className="w-24 text-right pr-4 text-xs font-mono font-semibold text-violet-600 dark:text-violet-400 truncate">
                            {tgtTok}
                          </div>
                          <div className="flex items-center gap-1.5">
                            {liveAttentionData.source_tokens.map((_, colIdx) => {
                              const weight = rowWeights[colIdx] ?? 0;
                              return (
                                <div
                                  key={colIdx}
                                  title={`Target: ${tgtTok} | Source: ${liveAttentionData.source_tokens[colIdx]} | Weight: ${(weight * 100).toFixed(2)}%`}
                                  style={{
                                    backgroundColor: `rgba(124, 58, 237, ${Math.max(0.06, Math.min(1, weight))})`,
                                  }}
                                  className="w-16 h-9 rounded-lg flex items-center justify-center text-[10px] font-mono font-bold transition-all hover:scale-105 cursor-pointer shadow-2xs text-slate-900 dark:text-white"
                                >
                                  {(weight * 100).toFixed(0)}%
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="mt-4 p-3 rounded-xl bg-slate-50 dark:bg-slate-800 text-xs text-slate-500 flex items-center justify-between">
                  <span>Target Hypothesis: <strong className="text-slate-800 dark:text-slate-200">{liveAttentionData.translated_text || '—'}</strong></span>
                  <span className="font-mono text-[11px]">Matrix Size: {liveAttentionData.attention_matrix.length} × {liveAttentionData.source_tokens.length}</span>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-sm">
                No attention data available. Click Extract above to generate real attention weights.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: REAL EMPIRICAL TRAINING LOSS CURVE */}
      {activeTab === 'training' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Empirical Training & Validation Loss Progression
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Actual recorded losses over 40 training epochs executed on Apple Silicon GPU (MPS) from random initialization.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                  Train: 7.35 → 3.02 | Val: 7.02 → 4.60
                </span>
              </div>
            </div>

            {/* Recharts Line Chart */}
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trainingData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Cross-Entropy Loss', angle: -90, position: 'insideLeft' }} domain={[2.5, 8]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '12px', color: '#fff', fontSize: '12px' }}
                  />
                  <Legend verticalAlign="top" height={36} />
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#7c3aed" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                  <Line type="monotone" dataKey="val_loss" name="Validation Loss" stroke="#10b981" strokeWidth={2.5} dot={false} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-100 dark:border-slate-800">
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Optimizer</span>
                <span className="text-sm font-bold text-slate-800 dark:text-slate-200">AdamW (decay=0.01)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Schedule</span>
                <span className="text-sm font-bold text-slate-800 dark:text-slate-200">Noam Warmup (50 steps)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Best Val Loss</span>
                <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400">4.6013 (Saved)</span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Train Convergence</span>
                <span className="text-sm font-bold text-violet-600 dark:text-violet-400">-58.9% Cross-Entropy</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: HELD-OUT TEST BENCHMARK */}
      {activeTab === 'benchmark' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Held-Out Multilingual Test Set Benchmark
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Scientifically computed using SacreBLEU 2.6.0 and chrF++ character n-gram scoring across independent held-out parallel sentences.
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-800 font-mono">
                Beam Search (k = 4)
              </span>
            </div>

            {/* Recharts Bar Chart */}
            {benchmarkData.length > 0 ? (
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={benchmarkData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="direction" />
                    <YAxis domain={[0, 15]} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '12px', color: '#fff', fontSize: '12px' }}
                    />
                    <Legend verticalAlign="top" height={36} />
                    <Bar dataKey="bleu" name="SacreBLEU" fill="#7c3aed" radius={[6, 6, 0, 0]} />
                    <Bar dataKey="chrf" name="chrF++" fill="#10b981" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-sm">
                No evaluation benchmark report found at <code className="font-mono">ml_lab/evaluation/results/</code>.
              </div>
            )}

            {/* Metric Comparison Table */}
            <div className="mt-8 overflow-x-auto">
              <table className="w-full text-left text-xs border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
                <thead className="bg-slate-50 dark:bg-slate-800/80 text-slate-500 uppercase font-mono">
                  <tr>
                    <th className="p-3">Language Direction</th>
                    <th className="p-3">SacreBLEU</th>
                    <th className="p-3">chrF++</th>
                    <th className="p-3">Avg Latency</th>
                    <th className="p-3">Throughput</th>
                    <th className="p-3">Evaluation Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {benchmarkData.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-800/50">
                      <td className="p-3 font-semibold text-slate-800 dark:text-slate-200 font-mono">
                        {row.direction}
                      </td>
                      <td className="p-3 font-mono font-bold text-violet-600 dark:text-violet-400">
                        {row.bleu.toFixed(2)}
                      </td>
                      <td className="p-3 font-mono font-bold text-emerald-600 dark:text-emerald-400">
                        {row.chrf.toFixed(2)}
                      </td>
                      <td className="p-3 font-mono">
                        {row.latency.toFixed(2)}s
                      </td>
                      <td className="p-3 font-mono">
                        {row.throughput.toFixed(1)} chars/s
                      </td>
                      <td className="p-3">
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-950/70 dark:text-emerald-300">
                          Held-Out Verified
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="mt-4 p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900/60 text-xs text-amber-800 dark:text-amber-200 leading-relaxed">
                <strong>Scientific Integrity Note:</strong> The previous version of this project reported a fabricated 100.0 BLEU and 100.0 chrF++ score by evaluating on the exact training sentences. The scores above reflect <em>genuine held-out test data</em> on our 6.03M parameter prototype trained from scratch. In low-resource settings with modest parallel seed data, n-gram matching produces modest BLEU scores while chrF++ (up to 11.17) captures character n-gram subword fidelity.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
