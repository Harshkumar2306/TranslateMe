import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, AlertTriangle, BookOpen, Layers, CheckCircle2, HardDrive, Sparkles } from 'lucide-react';
import { ModelInfo } from '../types';
import { fetchModelInfo } from '../services/api';

export const ModelInfoPage: React.FC = () => {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-950/60 border border-violet-200 dark:border-violet-800 text-violet-700 dark:text-violet-300 text-xs font-semibold mb-3">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Pure From-Scratch PyTorch Implementation</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Model Specifications & Technical Architecture
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Detailed technical specifications, mathematical foundations, and hardware configuration of the TranslateMe from-scratch Multilingual Seq2Seq Transformer.
        </p>
      </div>

      {/* Grid of Key Specs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-violet-50 dark:bg-violet-950/60 text-violet-600 dark:text-violet-400 flex items-center justify-center mb-3">
            <Cpu className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Parameters</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">
            {info?.num_parameters || '27,974,144 (27.97M)'}
          </span>
          <span className="text-[11px] text-slate-500">Trainable From Random Init</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-3">
            <Layers className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Vocabulary Size</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">8,192 Subwords</span>
          <span className="text-[11px] text-slate-500">ByteLevel Multilingual BPE Tokenizer</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-3">
            <HardDrive className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Device Accelerator</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">{info?.device || 'mps'}</span>
          <span className="text-[11px] text-slate-500">Apple Silicon MPS / CUDA</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Architecture</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">Seq2Seq Transformer</span>
          <span className="text-[11px] text-slate-500">Vaswani et al. (2017)</span>
        </div>
      </div>

      {/* Architecture Deep Dive */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <Layers className="w-5 h-5 text-violet-600 dark:text-violet-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Architectural Dimensions</h2>
          </div>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Encoder / Decoder Layers:</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">6 Encoder + 6 Decoder</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Embedding Dimension (d_model):</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">384</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Feedforward Dimension (d_ff):</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">1536</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Attention Heads:</span>
              <span className="font-mono text-violet-600 dark:text-violet-400 font-bold">6 Heads (d_k = d_v = 64)</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Weight Tying:</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">Encoder & Decoder & Projection Shared</span>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Training Pipeline & Optimization</h2>
          </div>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Optimizer:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">AdamW (β1=0.9, β2=0.98, decay=0.01)</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">LR Scheduler:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">Noam Warmup (50 steps)</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Loss Function:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">Cross-Entropy (label_smoothing=0.1)</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Gradient Clipping:</span>
              <span className="font-mono text-indigo-600 dark:text-indigo-400 font-bold">max_norm = 1.0</span>
            </div>
            <div className="flex justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60">
              <span className="font-semibold text-slate-800 dark:text-slate-200">Weight Initialization:</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400 font-bold">Xavier Uniform Linear, Normal Embeddings</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Limitations Card */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm mb-8">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Research Scope & Engineering Boundaries</h2>
        </div>
        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600 dark:text-slate-300">
          {(info?.limitations || [
            "Trained from scratch on curated 4-way parallel seed corpus",
            "Vocabulary size is 2,000 multilingual BPE subwords across 4 scripts",
            "Context length is optimized up to 128 subwords",
            "Zero pretrained weights: parameters randomly initialized with PyTorch"
          ]).map((lim, idx) => (
            <li key={idx} className="flex items-start gap-2 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 flex-shrink-0" />
              <span>{lim}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Mathematical Reference */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-5 h-5 text-violet-600 dark:text-violet-400" />
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Foundational Academic Reference</h2>
        </div>
        <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800 font-mono text-xs text-slate-700 dark:text-slate-300 leading-relaxed overflow-x-auto">
          Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. (2017). <br/>
          <strong>Attention Is All You Need.</strong> <br/>
          Advances in Neural Information Processing Systems 30 (NeurIPS 2017). arXiv:1706.03762.
        </div>
      </div>
    </div>
  );
};
