import React, { useState, useEffect } from 'react';
import {
  ArrowLeftRight,
  Copy,
  Check,
  Volume2,
  Trash2,
  Sparkles,
  Clock,
  AlertCircle,
  Wand2,
} from 'lucide-react';
import { Language, TranslationResult } from '../types';
import { translateText, detectLanguage } from '../services/api';
import { LanguageSelector } from '../components/LanguageSelector';

interface TranslatorPageProps {
  languages: Language[];
  initialSourceText?: string;
  initialSourceLang?: string;
  initialTargetLang?: string;
  onReuseConsumed?: () => void;
}

export const TranslatorPage: React.FC<TranslatorPageProps> = ({
  languages,
  initialSourceText = '',
  initialSourceLang = 'en',
  initialTargetLang = 'hi',
  onReuseConsumed,
}) => {
  const [sourceText, setSourceText] = useState(initialSourceText);
  const [sourceLang, setSourceLang] = useState(initialSourceLang);
  const [targetLang, setTargetLang] = useState(initialTargetLang);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [detectedBadge, setDetectedBadge] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [speakingSide, setSpeakingSide] = useState<'src' | 'tgt' | null>(null);

  useEffect(() => {
    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  useEffect(() => {
    if (initialSourceText) {
      setSourceText(initialSourceText);
      setSourceLang(initialSourceLang || 'en');
      setTargetLang(initialTargetLang || 'hi');
      setResult(null);
      setError(null);
      onReuseConsumed?.();
    }
  }, [initialSourceText, initialSourceLang, initialTargetLang, onReuseConsumed]);

  const isSameLanguage = sourceLang !== 'auto' && sourceLang === targetLang;

  const getLocale = (langCode: string): string => {
    const found = languages.find((l) => l.code === langCode);
    return found?.iso || langCode || 'en-US';
  };

  const handleSelectSource = (code: string) => {
    if (code !== 'auto' && code === targetLang) {
      if (sourceLang !== 'auto') {
        setTargetLang(sourceLang);
      } else {
        const alt = languages.find((l) => l.code !== code);
        if (alt) setTargetLang(alt.code);
      }
    }
    setSourceLang(code);
  };

  const handleSelectTarget = (code: string) => {
    if (code === sourceLang && sourceLang !== 'auto') {
      setSourceLang(targetLang);
    }
    setTargetLang(code);
  };

  const handleTranslate = async (
    overrideText?: string,
    overrideSrc?: string,
    overrideTgt?: string
  ) => {
    const textToRun = (overrideText ?? sourceText).trim();
    if (!textToRun) return;

    let effectiveSrc = overrideSrc ?? sourceLang;
    const effectiveTgt = overrideTgt ?? targetLang;

    if (effectiveSrc !== 'auto' && effectiveSrc === effectiveTgt) {
      setError('Source and target languages must be different.');
      return;
    }

    setLoading(true);
    setError(null);
    setDetectedBadge(null);

    try {
      if (effectiveSrc === 'auto') {
        const det = await detectLanguage(textToRun);
        effectiveSrc = det.detected_code || 'en';
        setDetectedBadge(`${det.detected_name || effectiveSrc} (${Math.round((det.confidence || 0.9) * 100)}%)`);
      }

      const res = await translateText(
        textToRun,
        effectiveSrc,
        effectiveTgt,
        2,
        false
      );
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSwap = () => {
    if (sourceLang === 'auto') return;
    const nextSrc = targetLang;
    const nextTgt = sourceLang;
    setSourceLang(nextSrc);
    setTargetLang(nextTgt);
    if (result?.translated_text) {
      setSourceText(result.translated_text);
      setResult(null);
    }
  };

  const handleCopy = (text: string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleSpeak = (text: string, langCode: string, side: 'src' | 'tgt') => {
    if (!('speechSynthesis' in window) || !text) return;
    if (speakingSide === side) {
      window.speechSynthesis.cancel();
      setSpeakingSide(null);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = getLocale(langCode);
    utterance.onstart = () => setSpeakingSide(side);
    utterance.onend = () => setSpeakingSide(null);
    utterance.onerror = () => setSpeakingSide(null);
    window.speechSynthesis.speak(utterance);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleTranslate();
    }
  };

  const wordCount = sourceText.trim() ? sourceText.trim().split(/\s+/).length : 0;

  const customPresets = [
    { label: 'EN → HI', src: 'en', tgt: 'hi', text: 'TranslateMe is an open-source multilingual neural machine translation engine.' },
    { label: 'EN → TA', src: 'en', tgt: 'ta', text: 'Machine learning models learn patterns from parallel training data.' },
    { label: 'EN → TE', src: 'en', tgt: 'te', text: 'Deep learning models use multi-layer neural networks.' },
    { label: 'HI → TA', src: 'hi', tgt: 'ta', text: 'नमस्ते, आज आप कैसे हैं?' },
    { label: 'TA → TE', src: 'ta', tgt: 'te', text: 'வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?' },
    { label: 'TE → EN', src: 'te', tgt: 'en', text: 'కంప్యూటర్ సైన్స్ ఆధునిక ప్రపంచాన్ని మారుస్తుంది.' },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-7 sm:py-9 animate-fade-up">
      {/* Hero Header */}
      <div className="text-center mb-7">
        <h1 className="text-3xl sm:text-4xl lg:text-[42px] font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
          TranslateMe Studio
        </h1>

        {/* Quick Examples */}
        <div className="mt-5 flex items-center justify-start sm:justify-center gap-2 overflow-x-auto no-scrollbar py-1 px-1">
          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider whitespace-nowrap mr-1 flex-shrink-0 flex items-center gap-1">
            <Wand2 className="w-3 h-3 text-brand-500" />
            Try:
          </span>
          {customPresets.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setSourceLang(p.src);
                setTargetLang(p.tgt);
                setSourceText(p.text);
                handleTranslate(p.text, p.src, p.tgt);
              }}
              className="px-3 py-1.5 text-xs font-semibold rounded-full border border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-300 hover:border-brand-500 hover:text-brand-600 dark:hover:text-brand-400 hover:bg-brand-50/40 dark:hover:bg-brand-950/30 transition-all shadow-2xs whitespace-nowrap flex-shrink-0 active:scale-95"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Translation Workspace Card */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200/90 dark:border-slate-800 shadow-xl shadow-slate-200/60 dark:shadow-black/30 overflow-visible">
        <div className="p-4 sm:p-6 border-b border-slate-100 dark:border-slate-800/80 bg-slate-50/70 dark:bg-slate-850/60 rounded-t-3xl grid grid-cols-1 sm:grid-cols-[1fr,auto,1fr] gap-3 sm:gap-4 items-center">
          <LanguageSelector
            label="Source Language"
            languages={languages}
            selectedCode={sourceLang}
            onSelect={handleSelectSource}
            allowAuto={true}
          />

          <div className="flex justify-center sm:pt-5 my-0.5 sm:my-0">
            <button
              type="button"
              onClick={handleSwap}
              disabled={sourceLang === 'auto'}
              title={sourceLang === 'auto' ? 'Select a specific source language to swap' : 'Swap source and target'}
              className="w-11 h-11 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-brand-50 dark:hover:bg-slate-700 hover:text-brand-600 hover:border-brand-300 active:rotate-180 active:scale-90 disabled:opacity-35 disabled:cursor-not-allowed transition-all duration-200 shadow-sm flex items-center justify-center"
            >
              <ArrowLeftRight className="w-4 h-4" />
            </button>
          </div>

          <LanguageSelector
            label="Target Language"
            languages={languages}
            selectedCode={targetLang}
            onSelect={handleSelectTarget}
            allowAuto={false}
          />
        </div>

        {/* Dual Pane Editor */}
        <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-100 dark:divide-slate-800">
          {/* Source Pane */}
          <div className="p-5 sm:p-6 flex flex-col justify-between min-h-[260px] sm:min-h-[290px]">
            <div className="flex-1">
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type or paste text in English, Hindi, Tamil, or Telugu..."
                className={`w-full h-44 sm:h-52 bg-transparent text-slate-800 dark:text-slate-100 placeholder:text-slate-400 resize-none focus:outline-none leading-relaxed font-sans ${
                  sourceText.length > 140 ? 'text-base' : 'text-lg sm:text-xl'
                }`}
              />
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-slate-800/70 text-xs text-slate-400">
              <div className="flex items-center gap-3">
                {sourceText && (
                  <button
                    type="button"
                    onClick={() => {
                      setSourceText('');
                      setResult(null);
                      setDetectedBadge(null);
                      setError(null);
                    }}
                    className="flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-rose-50 dark:hover:bg-rose-950/40 hover:text-rose-500 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Clear</span>
                  </button>
                )}
                {sourceText && 'speechSynthesis' in window && (
                  <button
                    type="button"
                    onClick={() =>
                      handleSpeak(
                        sourceText,
                        sourceLang === 'auto' ? (result?.source_language || 'en') : sourceLang,
                        'src'
                      )
                    }
                    className={`flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-brand-50 dark:hover:bg-brand-950/40 transition-colors ${
                      speakingSide === 'src' ? 'text-brand-600 font-semibold' : 'hover:text-brand-500'
                    }`}
                    title={speakingSide === 'src' ? 'Stop speaking' : 'Listen to source'}
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    <span>{speakingSide === 'src' ? 'Stop' : 'Listen'}</span>
                  </button>
                )}
              </div>

              <div className="flex items-center gap-2 font-mono text-[11px]">
                <span>{wordCount} words</span>
                <span>•</span>
                <span>{sourceText.length} chars</span>
              </div>
            </div>
          </div>

          {/* Target Output Pane */}
          <div className="p-5 sm:p-6 flex flex-col justify-between min-h-[260px] sm:min-h-[290px] bg-slate-50/45 dark:bg-slate-950/35">
            <div className="flex-1">
              {loading ? (
                <div className="h-44 sm:h-52 flex flex-col items-center justify-center gap-3 text-slate-400">
                  <div className="w-9 h-9 rounded-full border-2 border-brand-500 border-t-transparent animate-spin" />
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 animate-pulse">
                    Translating with Custom Transformer...
                  </p>
                </div>
              ) : error ? (
                <div className="h-44 sm:h-52 flex items-center justify-center p-2">
                  <div className="flex items-center gap-2.5 text-rose-600 dark:text-rose-400 text-xs sm:text-sm bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 p-4 rounded-2xl">
                    <AlertCircle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                  </div>
                </div>
              ) : result ? (
                <div className="h-44 sm:h-52 overflow-y-auto pr-2 animate-fade-up">
                  <p className={`text-slate-900 dark:text-white leading-relaxed font-sans select-all whitespace-pre-wrap ${
                    result.translated_text.length > 140 ? 'text-base' : 'text-lg sm:text-xl font-medium'
                  }`}>
                    {result.translated_text}
                  </p>
                </div>
              ) : (
                <div className="h-44 sm:h-52 flex flex-col items-center justify-center text-center text-slate-400 gap-2">
                  <Sparkles className="w-5 h-5 text-slate-300 dark:text-slate-700" />
                  <p className="text-sm">Translation output will appear here</p>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-slate-800/70 text-xs text-slate-400 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                {result && (
                  <>
                    <button
                      type="button"
                      onClick={() => handleCopy(result.translated_text)}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg hover:bg-white dark:hover:bg-slate-800 hover:text-brand-600 transition-colors font-semibold text-slate-600 dark:text-slate-300"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copied ? 'Copied!' : 'Copy'}</span>
                    </button>

                    {'speechSynthesis' in window && (
                      <button
                        type="button"
                        onClick={() => handleSpeak(result.translated_text, result.target_language || targetLang, 'tgt')}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg hover:bg-white dark:hover:bg-slate-800 transition-colors font-semibold ${
                          speakingSide === 'tgt' ? 'text-brand-600 dark:text-brand-400' : 'text-slate-600 dark:text-slate-300 hover:text-brand-500'
                        }`}
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                        <span>{speakingSide === 'tgt' ? 'Stop' : 'Listen'}</span>
                      </button>
                    )}
                  </>
                )}
              </div>

              {result && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {detectedBadge && (
                    <span className="bg-indigo-50 text-indigo-700 dark:bg-indigo-950/70 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 px-2 py-0.5 rounded-lg text-[11px] font-semibold">
                      Detected: {detectedBadge}
                    </span>
                  )}
                  <span className="flex items-center gap-1 bg-white dark:bg-slate-800 border border-slate-200/70 dark:border-slate-700 text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded-lg text-[11px] font-mono">
                    <Clock className="w-3 h-3 text-brand-500" />
                    {result.inference_time}s
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Action Toolbar */}
        <div className="p-4 sm:p-5 bg-slate-50/90 dark:bg-slate-850 rounded-b-3xl border-t border-slate-100 dark:border-slate-800 flex items-center justify-end gap-4">
          <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
            {isSameLanguage && (
              <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">
                Select two different languages
              </span>
            )}
            <span className="hidden lg:inline text-[11px] text-slate-400">
              <kbd className="px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-slate-700 font-mono text-[10px]">Ctrl+Enter</kbd>
            </span>
            <button
              type="button"
              onClick={() => handleTranslate()}
              disabled={loading || !sourceText.trim() || isSameLanguage}
              className="w-full sm:w-auto min-h-[46px] flex items-center justify-center gap-2 px-7 py-2.5 rounded-2xl bg-gradient-to-r from-brand-600 via-indigo-600 to-violet-600 hover:from-brand-700 hover:to-violet-700 text-white font-bold text-sm shadow-lg shadow-brand-500/25 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Translating...' : 'Translate'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
