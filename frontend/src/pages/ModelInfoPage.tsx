import React, { useEffect, useState } from 'react';
import { Cpu, ShieldCheck, AlertTriangle, BookOpen, Layers, CheckCircle2, HardDrive, GitBranch } from 'lucide-react';
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
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Model Specifications & Research Attribution
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          LinguaAI utilizes the open-source NLLB-200 (No Language Left Behind) Transformer developed by Meta AI Research, complemented by an educational from-scratch PyTorch Transformer.
        </p>
      </div>

      {/* Grid of Key Specs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 flex items-center justify-center mb-3">
            <Cpu className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Parameters</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">615 Million</span>
          <span className="text-[11px] text-slate-500">Distilled Seq2Seq Transformer</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-3">
            <Layers className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Vocabulary Size</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">256,204</span>
          <span className="text-[11px] text-slate-500">SentencePiece BPE Tokenizer</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-3">
            <HardDrive className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Device Accelerator</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">{info?.device || 'MPS / CUDA'}</span>
          <span className="text-[11px] text-slate-500">Apple Silicon / GPU Acceleration</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-9 h-9 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 flex items-center justify-center mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <span className="text-xs text-slate-400 block uppercase tracking-wider font-semibold">License</span>
          <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">CC-BY-NC 4.0</span>
          <span className="text-[11px] text-slate-500">Non-Commercial Research</span>
        </div>
      </div>

      {/* Model Attribution and Architecture Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen className="w-5 h-5 text-brand-600 dark:text-brand-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Academic Citation & Attribution</h2>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed mb-4">
            The primary translation model used in this system is <strong className="text-slate-800 dark:text-slate-100">facebook/nllb-200-distilled-600M</strong>. It is not represented as a model trained from scratch by this project.
          </p>
          <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 font-mono text-[11px] text-slate-700 dark:text-slate-300 leading-relaxed overflow-x-auto">
            NLLB Team, Marta R. Costa-jussà, et al. (2022). <br/>
            <strong>No Language Left Behind: Scaling Human-Centered Machine Translation.</strong> <br/>
            Meta AI Research. arXiv:2207.04672.
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">Two Distinct Transformer Engines</h2>
          </div>
          <div className="space-y-3 text-xs text-slate-600 dark:text-slate-300">
            <div className="p-3 rounded-xl bg-brand-50/50 dark:bg-brand-950/30 border border-brand-100 dark:border-brand-900">
              <span className="font-bold text-brand-700 dark:text-brand-300 block mb-0.5">Engine A: NLLB-200 (Meta AI)</span>
              <span>Production multilingual engine handling 202 languages, 615M weights, beam search, and subword tokenization.</span>
            </div>
            <div className="p-3 rounded-xl bg-indigo-50/50 dark:indigo-950/30 border border-indigo-100 dark:border-indigo-900">
              <span className="font-bold text-indigo-700 dark:text-indigo-300 block mb-0.5">Engine B: Custom PyTorch Transformer</span>
              <span>Educational implementation of 'Attention Is All You Need' with custom Scaled Dot-Product, Multi-Head Attention, and Positional Encodings.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Limitations Card */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-amber-500" />
          <h2 className="text-lg font-bold text-slate-900 dark:text-white">Research Limitations & Known Caveats</h2>
        </div>
        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-600 dark:text-slate-300">
          {info?.limitations.map((lim, idx) => (
            <li key={idx} className="flex items-start gap-2 p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 flex-shrink-0"></span>
              <span>{lim}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
