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
  ChevronDown,
  AlertCircle,
  Play
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
  initialSourceLang = 'eng_Latn',
  initialTargetLang = 'hin_Deva',
}) => {
  const [sourceText, setSourceText] = useState(initialSourceText);
  const [sourceLang, setSourceLang] = useState(initialSourceLang);
  const [targetLang, setTargetLang] = useState(initialTargetLang);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
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
        showAlternatives
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

  // Quick preset trials
  const presets = [
    { label: 'English → Hindi', src: 'eng_Latn', tgt: 'hin_Deva', text: 'Welcome to LinguaAI, an advanced Transformer neural machine translation system.' },
    { label: 'Hindi → Tamil', src: 'hin_Deva', tgt: 'tam_Taml', text: 'नमस्ते, आप कैसे हैं? आशा है कि आप अच्छे होंगे।' },
    { label: 'Tamil → French', src: 'tam_Taml', tgt: 'fra_Latn', text: 'வணக்கம், எப்படி இருக்கிறீர்கள்?' },
    { label: 'Telugu → Japanese', src: 'tel_Telu', tgt: 'jpn_Jpan', text: 'నమస్కారం, మీరు ఎలా ఉన్నారు?' },
    { label: 'Malayalam → Bengali', src: 'mal_Mlym', tgt: 'ben_Beng', text: 'നമസ്കാരം, സുഖമാണോ?' },
    { label: 'Spanish → English', src: 'spa_Latn', tgt: 'eng_Latn', text: 'La inteligencia artificial está transformando la investigación científica en todo el mundo.' },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Hero Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white">
          Universal Multilingual Translation
        </h1>
        <p className="mt-2 text-sm sm:text-base text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
          Translate seamlessly across hundreds of languages using the state-of-the-art NLLB-200 Transformer architecture. Direct many-to-many translation without English bridging.
        </p>

        {/* Quick Test Presets */}
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider mr-1">
            Quick Tests:
          </span>
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setSourceLang(p.src);
                setTargetLang(p.tgt);
                setSourceText(p.text);
              }}
              className="px-2.5 py-1 text-xs rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:border-brand-500 hover:text-brand-600 transition-colors shadow-2xs"
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
          <div className="flex justify-center sm:pt-5">
            <button
              type="button"
              onClick={handleSwap}
              disabled={sourceLang === 'auto'}
              title="Swap Languages"
              className="p-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 hover:text-brand-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-sm"
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
                <div className="flex items-center gap-2">
                  <span className="flex items-center gap-1 bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 px-2 py-0.5 rounded text-[11px] font-mono border border-brand-200 dark:border-brand-800">
                    <Clock className="w-3 h-3" />
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
          <div className="flex items-center gap-4 text-xs text-slate-500">
            {/* Beam search selector */}
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-slate-400" />
              <span>Beam Search:</span>
              <select
                value={numBeams}
                onChange={(e) => setNumBeams(Number(e.target.value))}
                className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-xs text-slate-700 dark:text-slate-200 focus:outline-none"
              >
                <option value={1}>1 (Greedy)</option>
                <option value={2}>2 Beams (Fast)</option>
                <option value={4}>4 Beams (Quality)</option>
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
          </div>

          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span className="hidden lg:inline text-[11px] text-slate-400">
              Press <kbd className="px-1.5 py-0.5 rounded bg-slate-200 dark:bg-slate-700 font-mono text-[10px]">Ctrl+Enter</kbd>
            </span>
            <button
              type="button"
              onClick={handleTranslate}
              disabled={loading || !sourceText.trim()}
              className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-700 hover:to-indigo-700 text-white font-semibold text-sm shadow-md shadow-brand-500/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
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
