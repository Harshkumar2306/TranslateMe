import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, AlertTriangle, BookOpen, Layers, CheckCircle2, HardDrive, Sparkles } from 'lucide-react';
import { ModelInfo } from '../types';
import { fetchModelInfo } from '../services/api';

interface ModelInfoPageProps {
  modelInfo?: ModelInfo | null;
}

export const ModelInfoPage: React.FC<ModelInfoPageProps> = ({ modelInfo: propInfo }) => {
  const [info, setInfo] = useState<ModelInfo | null>(propInfo || null);

  useEffect(() => {
    if (propInfo) {
      setInfo(propInfo);
      return;
    }
    fetchModelInfo().then(setInfo).catch(console.error);
  }, [propInfo]);

  if (!info) {
    return (
      <div className="max-w-5xl mx-auto px-4 py-8 space-y-4 animate-pulse">
        <div className="h-24 rounded-2xl bg-slate-200/60 dark:bg-slate-800/50" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-32 rounded-2xl bg-slate-200/60 dark:bg-slate-800/50" />
          ))}
        </div>
      </div>
    );
  }

  const encLayers = info.encoder_layers ?? info.layers;
  const decLayers = info.decoder_layers ?? info.layers;
  const dModel = info.d_model;
  const dFf = info.d_ff;
  const nHeads = info.attention_heads;
  const headDim = info.head_dim ?? (dModel && nHeads ? Math.round(dModel / nHeads) : undefined);
  const td = info.training_details;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 animate-fade-up">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-violet-700 dark:text-violet-300 text-xs font-semibold mb-3">
          <Sparkles className="w-3.5 h-3.5" />
          <span>From-Scratch PyTorch Architecture</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Model Specifications
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          {info.model_name} — live parameters, checkpoint metadata, and runtime device.
        </p>
      </div>

      {/* Key Specs Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-violet-50 dark:bg-violet-950/60 text-violet-600 dark:text-violet-400 flex items-center justify-center mb-3">
            <Cpu className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Parameters</span>
          <span className="text-lg font-bold text-slate-900 dark:text-white mt-1 block">
            {info.num_parameters}
          </span>
          <span className="text-[11px] text-slate-500">
            {info.checkpoint_size_mb ? `${info.checkpoint_size_mb} MB (${info.weight_dtype})` : 'From Random Init'}
          </span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-3">
            <Layers className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Vocabulary Size</span>
          <span className="text-lg font-bold text-slate-900 dark:text-white mt-1 block">
            {info.vocab_size}
          </span>
          <span className="text-[11px] text-slate-500 truncate block">{info.tokenizer_type}</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-3">
            <HardDrive className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Device Accelerator</span>
          <span className="text-lg font-bold text-slate-900 dark:text-white mt-1 block uppercase">
            {info.device}
          </span>
          <span className="text-[11px] text-slate-500">Local PyTorch Runtime</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Architecture</span>
          <span className="text-base font-bold text-slate-900 dark:text-white mt-1 block truncate" title={info.model_type}>
            {info.is_modern ? 'Modern RoPE + SwiGLU' : 'Seq2Seq Transformer'}
          </span>
          <span className="text-[11px] text-slate-500 truncate block">{info.framework}</span>
        </div>
      </div>

      {/* Architecture Deep Dive & Checkpoint State */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Active Model Dimensions</h2>
          </div>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Encoder / Decoder Layers:</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {encLayers} Enc / {decLayers} Dec
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Embedding Dimension (d_model):</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">{dModel}</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Feedforward Dimension (d_ff):</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">{dFf}</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Attention Heads:</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">
                {nHeads} Heads (d_k = {headDim})
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Weight Tying:</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                {info.weight_tying || 'Shared Embeddings'}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Loaded Checkpoint & Runtime</h2>
          </div>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Epochs / Global Steps:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                {td?.epochs_completed ?? info.epochs_completed ?? '—'} epochs • {td?.global_step?.toLocaleString() ?? info.global_step?.toLocaleString() ?? '—'} steps
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Best Validation Loss:</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                {info.best_val_loss != null ? info.best_val_loss.toFixed(4) : '—'}
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Max Context / Dropout:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                {info.max_len ?? '—'} tokens • p={info.dropout ?? '—'}
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Logit Soft-Cap / Norms:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">
                {info.soft_cap ? `tanh(${info.soft_cap})` : 'None'} • {info.use_qk_norm ? 'QK-Norm + RMSNorm' : 'LayerNorm'}
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Decoding Engine:</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">
                {td?.kv_cache_decoding || 'Autoregressive Decoding'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Limitations & Verified Details */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm mb-8">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">System Boundaries & Checkpoint Details</h2>
        </div>
        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600 dark:text-slate-300">
          {(info.limitations || []).map((lim, idx) => (
            <li key={idx} className="flex items-start gap-2 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
              <span>{lim}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Academic References */}
      {info.references && info.references.length > 0 && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Architectural References</h2>
          </div>
          <div className="space-y-2.5">
            {info.references.map((ref, idx) => (
              <div key={idx} className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 font-mono text-xs text-slate-700 dark:text-slate-300">
                <strong>{ref.title}</strong> — {ref.citation}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
