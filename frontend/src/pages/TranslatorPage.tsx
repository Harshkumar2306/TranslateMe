import React, { useState, useEffect } from 'react';
import {
  ArrowLeftRight,
  Copy,
  Check,
  Volume2,
  Trash2,
  Sparkles,
  Clock,
  Layers,
  AlertCircle,
  HelpCircle,
  Cpu,
  BrainCircuit
} from 'lucide-react';
import { Language, TranslationResult } from '../types';
import { translateText } from '../services/api';
import { LanguageSelector } from '../components/LanguageSelector';

interface TranslatorPageProps {
  languages: Language[];
  initialSourceText?: string;
  initialSourceLang?: string;
  initialTargetLang?: string;
}

export const TranslatorPage: React.FC<TranslatorPageProps> = ({
  languages,
  initialSourceText = '',
  initialSourceLang = 'en',
  initialTargetLang = 'hi',
}) => {
  const [sourceText, setSourceText] = useState(initialSourceText);
  const [sourceLang, setSourceLang] = useState(initialSourceLang);
  const [targetLang, setTargetLang] = useState(initialTargetLang);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedModel, setSelectedModel] = useState<'custom'>('custom');
  const [numBeams, setNumBeams] = useState(2);
  const [showAlternatives, setShowAlternatives] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Sync props if changed
  useEffect(() => {
    if (initialSourceText) setSourceText(initialSourceText);
    if (initialSourceLang) setSourceLang(initialSourceLang);
    if (initialTargetLang) setTargetLang(initialTargetLang);
  }, [initialSourceText, initialSourceLang, initialTargetLang]);

  // Handle translation
  const handleTranslate = async () => {
    if (!sourceText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await translateText(
        sourceText,
        sourceLang,
        targetLang,
        numBeams,
        showAlternatives,
        'custom'
      );
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  // Swap languages
  const handleSwap = () => {
    if (sourceLang === 'auto') return;
    const temp = sourceLang;
    setSourceLang(targetLang);
    setTargetLang(temp);
    if (result) {
      setSourceText(result.translated_text);
      setResult(null);
    }
  };

  // Copy to clipboard
  const handleCopy = () => {
    if (!result?.translated_text) return;
    navigator.clipboard.writeText(result.translated_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Text-to-Speech
  const handleSpeak = (text: string) => {
    if (!('speechSynthesis' in window) || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  // Keyboard shortcut: Cmd/Ctrl + Enter
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleTranslate();
    }
  };

  // Quick preset trials for the 4 supported languages
  const customPresets = [
    { label: 'English → Hindi', src: 'en', tgt: 'hi', text: 'TranslateMe is an open-source multilingual neural machine translation engine.' },
    { label: 'English → Tamil', src: 'en', tgt: 'ta', text: 'Machine learning models learn patterns from training data.' },
    { label: 'English → Telugu', src: 'en', tgt: 'te', text: 'Deep learning models use multi-layer neural networks.' },
    { label: 'Hindi → English', src: 'hi', tgt: 'en', text: 'नमस्ते, आप कैसे हैं?' },
    { label: 'Tamil → English', src: 'ta', tgt: 'en', text: 'செயற்கை நுண்ணறிவு மனிதர்களின் திறனை மேம்படுத்துகிறது.' },
    { label: 'Telugu → English', src: 'te', tgt: 'en', text: 'కంప్యూటర్ సైన్స్ ఆధునిక ప్రపంచాన్ని మారుస్తుంది.' },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Hero Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">
          From-Scratch Multilingual Transformer
        </h1>
        <p className="mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          100% custom PyTorch encoder-decoder Transformer. Randomly initialized, trained from scratch on parallel corpora with BPE subwords. Zero pretrained weights.
        </p>

        {/* Model Architecture Badges */}
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2.5">
          <div className="flex items-center gap-2 px-4 py-2 rounded-2xl text-xs sm:text-sm font-bold bg-gradient-to-r from-violet-600 to-indigo-600 text-white border border-violet-500 shadow-md shadow-violet-500/25">
            <Cpu className="w-4 h-4 text-violet-200" />
            <span>Custom Seq2Seq Transformer (Vaswani et al., 2017)</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-800/90 text-white font-extrabold uppercase tracking-wider">
              27.97M Params
            </span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 rounded-2xl text-xs font-semibold bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300">
            <Layers className="w-3.5 h-3.5 text-indigo-500" />
            <span>6 Enc / 6 Dec • 6 Heads • d_model=384 • 8,192 BPE Subwords</span>
          </div>
        </div>

        {/* Quick Test Presets */}
        <div className="mt-4 flex items-center justify-start sm:justify-center gap-2 overflow-x-auto no-scrollbar py-1 px-1">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap mr-1 flex-shrink-0">
            Parallel Presets:
          </span>
          {customPresets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setSourceLang(p.src);
                setTargetLang(p.tgt);
                setSourceText(p.text);
              }}
              className="px-3 py-1.5 text-xs font-medium rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:border-violet-500 hover:text-violet-600 transition-colors shadow-2xs whitespace-nowrap flex-shrink-0 active:scale-95"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Translation Card */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200/80 dark:border-slate-800 shadow-xl shadow-slate-200/50 dark:shadow-none overflow-hidden">
        {/* Controls Bar */}
        <div className="p-4 sm:p-6 border-b border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-800/20 grid grid-cols-1 sm:grid-cols-[1fr,auto,1fr] gap-4 items-center">
          <LanguageSelector
            label="Source Language"
            languages={languages}
            selectedCode={sourceLang}
            onSelect={setSourceLang}
            allowAuto={true}
          />

          {/* Swap button */}
          <div className="flex justify-center sm:pt-5 my-1 sm:my-0">
            <button
              type="button"
              onClick={handleSwap}
              disabled={sourceLang === 'auto'}
              title="Swap Languages"
              className="w-11 h-11 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-brand-600 active:rotate-180 active:scale-90 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-sm flex items-center justify-center"
            >
              <ArrowLeftRight className="w-4 h-4" />
            </button>
          </div>

          <LanguageSelector
            label="Target Language"
            languages={languages}
            selectedCode={targetLang}
            onSelect={setTargetLang}
            allowAuto={false}
          />
        </div>

        {/* Text Areas Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-100 dark:divide-slate-800">
          {/* Source Text Box */}
          <div className="p-5 sm:p-6 flex flex-col justify-between min-h-[280px]">
            <div className="flex-1">
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type, paste, or select a sample sentence..."
                className="w-full h-44 sm:h-52 bg-transparent text-slate-800 dark:text-slate-100 placeholder:text-slate-400 resize-none focus:outline-none text-base sm:text-lg leading-relaxed font-sans"
              />
            </div>

            {/* Source Footer Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-slate-800/60 text-xs text-slate-400">
              <div className="flex items-center gap-3">
                {sourceText && (
                  <button
                    onClick={() => {
                      setSourceText('');
                      setResult(null);
                    }}
                    className="flex items-center gap-1 hover:text-rose-500 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                )}
                {sourceText && (
                  <button
                    onClick={() => handleSpeak(sourceText)}
                    className="flex items-center gap-1 hover:text-brand-500 transition-colors"
                    title="Listen to source"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              <div className="flex items-center gap-3">
                <span>{sourceText.length} characters</span>
              </div>
            </div>
          </div>

          {/* Target Text Box */}
          <div className="p-5 sm:p-6 flex flex-col justify-between min-h-[280px] bg-slate-50/30 dark:bg-slate-900/40">
            <div className="flex-1">
              {loading ? (
                <div className="h-44 sm:h-52 flex flex-col items-center justify-center gap-3 text-slate-400">
                  <div className="w-8 h-8 rounded-full border-3 border-brand-500 border-t-transparent animate-spin"></div>
                  <p className="text-xs font-medium animate-pulse">Running Transformer Inference...</p>
                </div>
              ) : error ? (
                <div className="h-44 sm:h-52 flex items-center justify-center p-4">
                  <div className="flex items-center gap-2.5 text-rose-500 text-xs sm:text-sm bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 p-4 rounded-xl">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </div>
                </div>
              ) : result ? (
                <div className="h-44 sm:h-52 overflow-y-auto pr-2">
                  <p className="text-slate-900 dark:text-white text-base sm:text-lg leading-relaxed font-sans select-all whitespace-pre-wrap">
                    {result.translated_text}
                  </p>
                </div>
              ) : (
                <div className="h-44 sm:h-52 flex items-center justify-center text-slate-400 text-sm">
                  Translation will appear here...
                </div>
              )}
            </div>

            {/* Target Footer Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-slate-800/60 text-xs text-slate-400">
              <div className="flex items-center gap-3">
                {result && (
                  <>
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1.5 hover:text-brand-500 transition-colors font-medium text-slate-600 dark:text-slate-300"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Copied!' : 'Copy'}</span>
                    </button>

                    <button
                      onClick={() => handleSpeak(result.translated_text)}
                      className={`flex items-center gap-1.5 hover:text-brand-500 transition-colors font-medium ${
                        isSpeaking ? 'text-brand-500' : 'text-slate-600 dark:text-slate-300'
                      }`}
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                      <span>Listen</span>
                    </button>
                  </>
                )}
              </div>

              {result && (
                <div className="flex flex-wrap items-center gap-2">
                  <span className="bg-violet-100 text-violet-700 dark:bg-violet-950/80 dark:text-violet-300 border border-violet-300 dark:border-violet-800 px-2 py-0.5 rounded text-[11px] font-bold">
                    Custom PyTorch Transformer
                  </span>
                  <span className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded text-[11px] font-mono">
                    <Clock className="w-3 h-3 text-brand-500" />
                    {result.inference_time}s
                  </span>
                  {result.detected_language && (
                    <span className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 px-2 py-0.5 rounded text-[11px]">
                      Detected: {result.source_language_name}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Translation Submit Bar */}
        <div className="p-4 sm:p-5 bg-slate-50 dark:bg-slate-850 border-t border-slate-100 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
            {/* Beam search selector */}
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-400" />
              <span>Decoding:</span>
              <select
                value={numBeams}
                onChange={(e) => setNumBeams(Number(e.target.value))}
                className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-700 dark:text-slate-200 focus:outline-none"
              >
                <option value={1}>Greedy (Fastest)</option>
                <option value={2}>2 Beams (Balanced)</option>
                <option value={4}>4 Beams (Highest Quality)</option>
              </select>
            </div>

            {/* Alternatives toggle */}
            <label className="hidden sm:flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showAlternatives}
                onChange={(e) => setShowAlternatives(e.target.checked)}
                className="rounded text-brand-600 focus:ring-brand-500 dark:bg-slate-800"
              />
              <span>Generate Alternatives</span>
            </label>

            <span className="hidden xl:inline-flex items-center gap-1 text-[11px] text-slate-400">
              <HelpCircle className="w-3.5 h-3.5 text-brand-500" />
              <span>Automatic Truecasing & anti-hallucination enabled</span>
            </span>
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span className="hidden lg:inline text-[11px] text-slate-400">
              Press <kbd className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 font-mono text-[10px]">Ctrl+Enter</kbd>
            </span>
            <button
              type="button"
              onClick={handleTranslate}
              disabled={loading || !sourceText.trim()}
              className="w-full sm:w-auto min-h-[48px] flex items-center justify-center gap-2 px-7 py-3 rounded-2xl bg-gradient-to-r from-brand-600 via-indigo-600 to-brand-700 hover:from-brand-700 hover:to-indigo-800 text-white font-bold text-sm shadow-lg shadow-brand-500/25 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Translating...' : 'Translate'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Alternative Hypotheses Section */}
      {result && result.alternatives && result.alternatives.length > 0 && (
        <div className="mt-6 p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            <Layers className="w-4 h-4 text-brand-500" />
            <span>Alternative Beam Hypotheses</span>
          </div>
          <div className="space-y-2">
            {result.alternatives.map((alt, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 text-sm text-slate-800 dark:text-slate-200"
              >
                <span>{alt}</span>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(alt);
                  }}
                  className="p-1 hover:text-brand-500 transition-colors"
                  title="Copy alternative"
                >
                  <Copy className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
