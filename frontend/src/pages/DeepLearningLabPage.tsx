import React, { useEffect, useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import {
  Layers, Activity, Sparkles, BookOpen, Award, RefreshCw, Send, AlertCircle, ChevronDown, ChevronUp,
} from 'lucide-react';
import { LabInfo, ModelInfo, Language, AttentionResult } from '../types';
import { fetchLabInfo, fetchAttentionMatrix } from '../services/api';

interface DeepLearningLabPageProps {
  modelInfo?: ModelInfo | null;
  languages?: Language[];
}

export const DeepLearningLabPage: React.FC<DeepLearningLabPageProps> = ({ modelInfo, languages = [] }) => {
  const [labInfo, setLabInfo] = useState<LabInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [labError, setLabError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'architecture' | 'attention' | 'training' | 'benchmark'>('architecture');
  const [hoverCell, setHoverCell] = useState<{ r: number; c: number } | null>(null);
  const [expandedDir, setExpandedDir] = useState<string | null>(null);

  // Interactive Live Attention State
  const [attnInputText, setAttnInputText] = useState('Hello, how are you today?');
  const [attnSrcLang, setAttnSrcLang] = useState('en');
  const [attnTgtLang, setAttnTgtLang] = useState('hi');
  const [attnLoading, setAttnLoading] = useState(false);
  const [attnError, setAttnError] = useState<string | null>(null);
  const [liveAttentionData, setLiveAttentionData] = useState<AttentionResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchLabInfo()
      .then((data) => {
        if (!cancelled) setLabInfo(data);
      })
      .catch((err) => {
        if (!cancelled) setLabError(err.message || 'Failed to load lab data');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (activeTab === 'attention' && !liveAttentionData && !attnLoading && !attnError) {
      handleFetchAttention(attnInputText, attnSrcLang, attnTgtLang);
    }
  }, [activeTab]);

  const handleFetchAttention = async (text: string, src: string, tgt: string) => {
    if (!text.trim()) return;
    if (src === tgt) {
      setAttnError('Source and target languages must be different.');
      return;
    }
    setAttnLoading(true);
    setAttnError(null);
    try {
      const data = await fetchAttentionMatrix(text, src, tgt);
      setLiveAttentionData(data);
    } catch (err: any) {
      setAttnError(err.message || 'Failed to extract attention weights');
    } finally {
      setAttnLoading(false);
    }
  };

  const ct = labInfo?.custom_transformer;
  const encLayers = ct?.encoder_layers ?? modelInfo?.encoder_layers ?? modelInfo?.layers;
  const decLayers = ct?.decoder_layers ?? modelInfo?.decoder_layers ?? modelInfo?.layers;
  const dModel = ct?.d_model ?? modelInfo?.d_model;
  const dFf = ct?.d_ff ?? modelInfo?.d_ff;
  const headsCount = ct?.attention_heads ?? modelInfo?.attention_heads;
  const headDim = ct?.head_dim ?? (dModel && headsCount ? Math.round(dModel / headsCount) : undefined);
  const vocabCount = ct?.shared_vocab_size;
  const isModern = ct?.is_modern ?? modelInfo?.is_modern ?? false;

  const trainingCurve = ct?.training_curve ?? [];
  const trainingSummary = ct?.training_summary;
  const trainingXLabel = ct?.training_x_label || 'Step';

  const benchmarkObj = labInfo?.evaluation_benchmark || {};
  const benchmarkData = Object.keys(benchmarkObj).map((key) => {
    const item = benchmarkObj[key];
    return {
      key,
      direction: item.direction || key,
      bleu: Number((item.bleu || 0).toFixed(2)),
      chrf: Number((item.chrf || 0).toFixed(2)),
      latency: Number((item.avg_latency_sec || 0).toFixed(3)),
      throughput: Number((item.throughput_chars_per_sec || 0).toFixed(1)),
      emptyRate: item.empty_rate ?? 0,
      repetitionRate: item.repetition_rate ?? 0,
      samples: item.samples ?? [],
    };
  });
  const evalSummary = labInfo?.evaluation_summary;

  const langOptions = languages.length > 0
    ? languages
    : [
        { code: 'en', name: 'English' },
        { code: 'hi', name: 'Hindi' },
        { code: 'ta', name: 'Tamil' },
        { code: 'te', name: 'Telugu' },
      ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-violet-700 dark:text-violet-300 text-xs font-semibold mb-3">
          <Sparkles className="w-3.5 h-3.5" />
          <span>PyTorch Transformer Research Lab</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-white">
          Deep Learning Laboratory
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Live cross-attention inspection, real checkpoint loss trajectories, and held-out benchmark scores from the loaded model.
        </p>
      </div>

      {labError && (
        <div className="mb-6 p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{labError}</span>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex justify-center mb-8 overflow-x-auto no-scrollbar py-1">
        <div className="inline-flex p-1 rounded-2xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 max-w-full overflow-x-auto no-scrollbar">
          {[
            { id: 'architecture', label: 'Architecture & Equations', icon: BookOpen },
            { id: 'attention', label: 'Live Attention Heatmap', icon: Layers },
            { id: 'training', label: 'Training Curves', icon: Activity },
            { id: 'benchmark', label: 'Held-Out Benchmark', icon: Award },
          ].map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
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

      {/* TAB 1: ARCHITECTURE & EQUATIONS */}
      {activeTab === 'architecture' && (
        <div className="space-y-8 animate-in fade-in duration-200">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Active Params</span>
              <span className="text-lg sm:text-xl font-extrabold text-violet-600 dark:text-violet-400">
                {ct ? ct.parameters.toLocaleString() : loading ? '...' : '—'}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">
                {ct ? `${(ct.parameters / 1e6).toFixed(2)}M (${ct.weight_dtype || 'fp32'})` : 'From Scratch'}
              </span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Layer Depth</span>
              <span className="text-lg sm:text-xl font-extrabold text-slate-800 dark:text-slate-200">
                {encLayers && decLayers ? `${encLayers} Enc / ${decLayers} Dec` : loading ? '...' : '—'}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">
                {isModern ? 'Modern Asymmetric Stack' : 'Encoder-Decoder Stack'}
              </span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Dimensions</span>
              <span className="text-lg sm:text-xl font-extrabold text-slate-800 dark:text-slate-200">
                {dModel && dFf ? `d=${dModel}, ff=${dFf}` : loading ? '...' : '—'}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">
                {headsCount && headDim ? `${headsCount} Heads (d_k = ${headDim})` : ''}
              </span>
            </div>
            <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center shadow-xs">
              <span className="text-xs text-slate-400 block uppercase font-mono">Tokenizer</span>
              <span className="text-lg sm:text-xl font-extrabold text-emerald-600 dark:text-emerald-400">
                {vocabCount ? `${vocabCount.toLocaleString()} BPE` : loading ? '...' : '—'}
              </span>
              <span className="text-[10px] text-slate-500 block mt-0.5">
                {ct?.share_embeddings ? '3-Way Shared Embeddings' : 'ByteLevel Subwords'}
              </span>
            </div>
          </div>

          {/* Architectural Diagram Box */}
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <h2 className="text-xl font-bold text-slate-900 dark:text-white mb-2">
              {isModern ? 'Modern Seq2Seq Transformer Architecture' : 'Encoder-Decoder Transformer Architecture'}
            </h2>
            <p className="text-xs text-slate-500 mb-6">
              Live topology read from <code className="font-mono text-violet-600">{ct?.name || 'loaded PyTorch checkpoint'}</code> on device <code className="font-mono">{ct?.device || 'cpu'}</code>.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Encoder Stack */}
              <div className="p-5 rounded-2xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/60">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-bold text-sm text-blue-700 dark:text-blue-300 uppercase tracking-wider">
                    Encoder Stack
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-mono">
                    N = {encLayers ?? '—'} Layers
                  </span>
                </div>

                <div className="space-y-2.5 text-xs font-mono">
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-blue-100 dark:border-blue-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    {isModern
                      ? `SwiGLU Gated FFN (d=${dModel} → d_ff=${dFf} → ${dModel})`
                      : `Position-Wise FFN (d=${dModel} → d_ff=${dFf} → ${dModel})`}
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">
                    ▲ {isModern ? 'Pre-LN RMSNorm + LayerScale Residual' : 'Residual Add & LayerNorm'}
                  </div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-blue-100 dark:border-blue-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Multi-Head Self-Attention (h={headsCount ?? '—'}, d_k={headDim ?? '—'}
                    {ct?.use_qk_norm ? ', QK-Norm' : ''}
                    {ct?.use_rope ? ', RoPE' : ''})
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">
                    ▲ {isModern ? 'Pre-LN RMSNorm + LayerScale Residual' : 'Residual Add & LayerNorm'}
                  </div>
                  <div className="p-2.5 rounded-xl bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 text-center">
                    {isModern
                      ? `Shared Subword Embeddings (scaled by √${dModel ?? 'd'})`
                      : `Subword Embeddings (scaled by √${dModel ?? 'd'}) + Sinusoidal PE`}
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">▲ Source Token IDs + Padding Mask</div>
                </div>
              </div>

              {/* Decoder Stack */}
              <div className="p-5 rounded-2xl bg-purple-50/50 dark:bg-purple-950/20 border border-purple-200 dark:border-purple-900/60">
                <div className="flex items-center justify-between mb-4">
                  <span className="font-bold text-sm text-purple-700 dark:text-purple-300 uppercase tracking-wider">
                    Decoder Stack
                  </span>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 font-mono">
                    N = {decLayers ?? '—'} Layers (KV-Cached)
                  </span>
                </div>

                <div className="space-y-2.5 text-xs font-mono">
                  <div className="p-2.5 rounded-xl bg-purple-100 dark:bg-purple-900/50 text-purple-800 dark:text-purple-200 text-center">
                    Tied LM Head{ct?.soft_cap ? ` + tanh Soft-Cap (${ct.soft_cap})` : ''} (Vocab: {vocabCount?.toLocaleString() ?? '—'})
                  </div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-purple-100 dark:border-purple-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    {isModern ? `SwiGLU Gated FFN (d_ff=${dFf})` : `Position-Wise FFN (d_ff=${dFf})`}
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">
                    ▲ {isModern ? 'Pre-LN RMSNorm + LayerScale Residual' : 'Residual Add & LayerNorm'}
                  </div>
                  <div className="p-2.5 rounded-xl bg-violet-100 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-center text-violet-800 dark:text-violet-200 font-semibold shadow-2xs">
                    Multi-Head Cross-Attention (Query: Decoder, K/V: Encoder Memory)
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">
                    ▲ {isModern ? 'Pre-LN RMSNorm + LayerScale Residual' : 'Residual Add & LayerNorm'}
                  </div>
                  <div className="p-2.5 rounded-xl bg-white dark:bg-slate-800 border border-purple-100 dark:border-purple-900 text-center text-slate-700 dark:text-slate-200 shadow-2xs">
                    Causal Self-Attention{ct?.use_rope ? ' + RoPE' : ''}{ct?.use_qk_norm ? ' + QK-Norm' : ''}
                  </div>
                  <div className="text-center text-slate-400 text-[10px]">
                    ▲ Target Shifted Right + Direction Control Token (&lt;2lang&gt;)
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Mathematical Formulations from API */}
          {labInfo?.transformer_components && labInfo.transformer_components.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {labInfo.transformer_components.map((comp, idx) => (
                <div key={idx} className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800">
                  <h3 className="font-bold text-slate-900 dark:text-white text-base mb-2">
                    {idx + 1}. {comp.name}
                  </h3>
                  <p className="text-xs text-slate-500 mb-3 leading-relaxed">
                    {comp.description}
                  </p>
                  <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-xs text-violet-600 dark:text-violet-400 overflow-x-auto">
                    {comp.formula}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB 2: LIVE ATTENTION MATRIX */}
      {activeTab === 'attention' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Live Cross-Attention Heatmap
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Actual attention weights extracted live during PyTorch decoding.
                </p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-800 font-mono">
                Layer {liveAttentionData?.decoder_layer ?? decLayers ?? '—'} Cross-Attention (Mean Across {liveAttentionData?.num_heads ?? headsCount ?? '—'} Heads)
              </span>
            </div>

            {/* Interactive Controller */}
            <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 mb-6">
              <div className="flex flex-col sm:flex-row gap-3 items-center">
                <input
                  type="text"
                  value={attnInputText}
                  onChange={(e) => setAttnInputText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleFetchAttention(attnInputText, attnSrcLang, attnTgtLang);
                    }
                  }}
                  placeholder="Enter sentence to extract attention..."
                  className="flex-1 w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
                <div className="flex items-center gap-2 w-full sm:w-auto">
                  <select
                    value={attnSrcLang}
                    onChange={(e) => {
                      const next = e.target.value;
                      if (next === attnTgtLang) setAttnTgtLang(attnSrcLang);
                      setAttnSrcLang(next);
                    }}
                    className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 dark:text-slate-300"
                  >
                    {langOptions.map((l) => (
                      <option key={l.code} value={l.code}>{l.name} ({l.code})</option>
                    ))}
                  </select>
                  <span className="text-xs text-slate-400">→</span>
                  <select
                    value={attnTgtLang}
                    onChange={(e) => {
                      const next = e.target.value;
                      if (next === attnSrcLang) setAttnSrcLang(attnTgtLang);
                      setAttnTgtLang(next);
                    }}
                    className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 dark:text-slate-300"
                  >
                    {langOptions.map((l) => (
                      <option key={l.code} value={l.code}>{l.name} ({l.code})</option>
                    ))}
                  </select>
                  <button
                    type="button"
                    onClick={() => handleFetchAttention(attnInputText, attnSrcLang, attnTgtLang)}
                    disabled={attnLoading || !attnInputText.trim() || attnSrcLang === attnTgtLang}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-700 text-white text-xs font-bold transition-all disabled:opacity-50"
                  >
                    {attnLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
                    <span>Extract</span>
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2 mt-3 overflow-x-auto no-scrollbar pt-1">
                <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
                  Try:
                </span>
                {[
                  { text: 'Hello, how are you today?', src: 'en', tgt: 'hi' },
                  { text: 'Machine learning models learn patterns.', src: 'en', tgt: 'ta' },
                  { text: 'Deep learning requires neural networks.', src: 'en', tgt: 'te' },
                  { text: 'नमस्ते, आप कैसे हैं?', src: 'hi', tgt: 'en' },
                ].map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
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
                  <div className="flex items-center gap-1.5 ml-28 mb-2">
                    {liveAttentionData.source_tokens.map((tok, i) => (
                      <div
                        key={i}
                        title={liveAttentionData.raw_source_tokens?.[i] || tok}
                        className={`w-16 text-center text-[11px] font-mono font-semibold truncate px-1 transition-colors ${
                          hoverCell?.c === i ? 'text-brand-600 dark:text-brand-400 font-bold' : 'text-slate-600 dark:text-slate-300'
                        }`}
                      >
                        {tok}
                      </div>
                    ))}
                  </div>

                  <div className="space-y-1.5" onMouseLeave={() => setHoverCell(null)}>
                    {liveAttentionData.target_tokens.map((tgtTok, rowIdx) => {
                      const rowWeights = liveAttentionData.attention_matrix[rowIdx] || [];
                      return (
                        <div key={rowIdx} className="flex items-center">
                          <div
                            title={liveAttentionData.raw_target_tokens?.[rowIdx] || tgtTok}
                            className={`w-28 text-right pr-4 text-xs font-mono font-semibold truncate transition-colors ${
                              hoverCell?.r === rowIdx ? 'text-brand-600 dark:text-brand-400 font-bold' : 'text-violet-600 dark:text-violet-400'
                            }`}
                          >
                            {tgtTok}
                          </div>
                          <div className="flex items-center gap-1.5">
                            {liveAttentionData.source_tokens.map((srcTok, colIdx) => {
                              const weight = rowWeights[colIdx] ?? 0;
                              const isHighlighted = hoverCell?.r === rowIdx || hoverCell?.c === colIdx;
                              return (
                                <div
                                  key={colIdx}
                                  onMouseEnter={() => setHoverCell({ r: rowIdx, c: colIdx })}
                                  title={`Target: ${tgtTok} | Source: ${srcTok} | Weight: ${(weight * 100).toFixed(2)}%`}
                                  style={{
                                    backgroundColor: `rgba(124, 58, 237, ${Math.max(0.06, Math.min(1, weight))})`,
                                  }}
                                  className={`w-16 h-9 rounded-lg flex items-center justify-center text-[10px] font-mono font-bold transition-all cursor-pointer shadow-2xs ${
                                    weight > 0.45 ? 'text-white' : 'text-slate-900 dark:text-white'
                                  } ${isHighlighted ? 'ring-2 ring-brand-500/70 scale-105' : ''}`}
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

                <div className="mt-4 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-xs text-slate-500 flex flex-wrap items-center justify-between gap-2">
                  <span>Target Hypothesis: <strong className="text-slate-800 dark:text-slate-200">{liveAttentionData.translated_text || '—'}</strong></span>
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5 text-[11px]">
                      <span>0%</span>
                      <span className="inline-block w-14 h-2 rounded-full bg-gradient-to-r from-violet-100 via-violet-400 to-violet-700" />
                      <span>100%</span>
                    </span>
                    <span className="font-mono text-[11px]">Matrix: {liveAttentionData.attention_matrix.length} × {liveAttentionData.source_tokens.length}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-sm">
                {attnLoading ? 'Extracting cross-attention weights...' : 'No attention data available.'}
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: REAL CHECKPOINT TRAINING LOSS CURVES */}
      {activeTab === 'training' && (
        <div className="space-y-6 animate-in fade-in duration-200">
          <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 sm:p-8 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-bold text-slate-900 dark:text-white">
                  Checkpoint Training & Validation Loss Curves
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Loaded directly from active checkpoint ({trainingSummary?.train_points_count?.toLocaleString() ?? 0} train logs, {trainingSummary?.val_points_count ?? 0} validation snapshots).
                </p>
              </div>
              {trainingSummary?.start_train_loss != null && trainingSummary?.end_train_loss != null && (
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-xs font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                    Train: {trainingSummary.start_train_loss.toFixed(2)} → {trainingSummary.end_train_loss.toFixed(2)}
                    {trainingSummary.start_val_loss != null && trainingSummary.best_val_loss != null
                      ? ` | Val: ${trainingSummary.start_val_loss.toFixed(2)} → ${trainingSummary.best_val_loss.toFixed(2)}`
                      : ''}
                  </span>
                </div>
              )}
            </div>

            {trainingCurve.length > 0 ? (
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingCurve}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis
                      dataKey="step"
                      type="number"
                      domain={['dataMin', 'dataMax']}
                      tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))}
                      label={{ value: trainingXLabel, position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      domain={['auto', 'auto']}
                      label={{ value: 'Cross-Entropy Loss', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '12px', color: '#fff', fontSize: '12px' }}
                      labelFormatter={(label) => `${trainingXLabel}: ${Number(label).toLocaleString()}`}
                    />
                    <Legend verticalAlign="top" height={36} />
                    <Line
                      type="monotone"
                      dataKey="train_loss"
                      name="Train Loss"
                      stroke="#7c3aed"
                      strokeWidth={2}
                      dot={false}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="val_loss"
                      name="Validation Loss"
                      stroke="#10b981"
                      strokeWidth={2.5}
                      dot={false}
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="py-12 text-center text-slate-400 text-sm">
                No training history embedded in active checkpoint.
              </div>
            )}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6 pt-6 border-t border-slate-100 dark:border-slate-800">
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Epochs Completed</span>
                <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                  {ct?.epochs_completed ?? '—'}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Optimizer Steps</span>
                <span className="text-sm font-bold text-slate-800 dark:text-slate-200 font-mono">
                  {ct?.global_step != null ? ct.global_step.toLocaleString() : '—'}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Best Val Loss</span>
                <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                  {ct?.best_val_loss != null ? ct.best_val_loss.toFixed(4) : '—'}
                </span>
              </div>
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                <span className="text-[11px] text-slate-400 block uppercase">Train Loss Reduction</span>
                <span className="text-sm font-bold text-violet-600 dark:text-violet-400 font-mono">
                  {trainingSummary?.loss_reduction_pct != null ? `-${trainingSummary.loss_reduction_pct}%` : '—'}
                </span>
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
                  Held-Out Multilingual Test Benchmark
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  {evalSummary
                    ? `${evalSummary.dataset_source} • ${evalSummary.directions_count} directions • Mean BLEU ${evalSummary.avg_bleu} • Mean chrF++ ${evalSummary.avg_chrf}`
                    : 'SacreBLEU and chrF++ evaluation across held-out parallel sentences.'}
                </p>
              </div>
              {evalSummary?.decoding_algorithm && (
                <span className="text-xs px-2.5 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 border border-violet-200 dark:border-violet-800 font-mono">
                  {evalSummary.decoding_algorithm}
                </span>
              )}
            </div>

            {benchmarkData.length > 0 ? (
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={benchmarkData}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                    <XAxis dataKey="direction" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 'auto']} label={{ value: 'Score', angle: -90, position: 'insideLeft' }} />
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

            {benchmarkData.length > 0 && (
              <div className="mt-8 overflow-x-auto">
                <table className="w-full text-left text-xs border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
                  <thead className="bg-slate-50 dark:bg-slate-800/80 text-slate-500 uppercase font-mono">
                    <tr>
                      <th className="p-3">Direction</th>
                      <th className="p-3">SacreBLEU</th>
                      <th className="p-3">chrF++</th>
                      <th className="p-3">Avg Latency</th>
                      <th className="p-3">Throughput</th>
                      <th className="p-3">Samples</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                    {benchmarkData.map((row) => (
                      <React.Fragment key={row.key}>
                        <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-800/50">
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
                            {row.latency.toFixed(3)}s
                          </td>
                          <td className="p-3 font-mono">
                            {row.throughput.toFixed(1)} chars/s
                          </td>
                          <td className="p-3">
                            {row.samples.length > 0 && (
                              <button
                                type="button"
                                onClick={() => setExpandedDir(expandedDir === row.key ? null : row.key)}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-violet-50 dark:hover:bg-violet-950/50 text-slate-700 dark:text-slate-300 font-semibold"
                              >
                                <span>{expandedDir === row.key ? 'Hide' : 'Inspect'}</span>
                                {expandedDir === row.key ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                              </button>
                            )}
                          </td>
                        </tr>
                        {expandedDir === row.key && (
                          <tr>
                            <td colSpan={6} className="p-4 bg-slate-50/80 dark:bg-slate-950/50">
                              <div className="space-y-3">
                                {row.samples.map((samp, sIdx) => (
                                  <div key={sIdx} className="p-3 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1 text-xs">
                                    <div><span className="font-semibold text-slate-400">SRC:</span> <span className="text-slate-800 dark:text-slate-200">{samp.source}</span></div>
                                    <div><span className="font-semibold text-emerald-600 dark:text-emerald-400">REF:</span> <span className="text-slate-700 dark:text-slate-300">{samp.reference}</span></div>
                                    <div><span className="font-semibold text-violet-600 dark:text-violet-400">HYP:</span> <span className="text-slate-900 dark:text-white font-medium">{samp.hypothesis}</span></div>
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
