import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, Sparkles, AlertCircle, ArrowLeftRight, Copy, Check } from 'lucide-react';
import { Language, TranslationResult } from '../types';
import { translateText } from '../services/api';
import { LanguageSelector } from '../components/LanguageSelector';

interface VoiceTranslatorPageProps {
  languages: Language[];
}

export const VoiceTranslatorPage: React.FC<VoiceTranslatorPageProps> = ({ languages }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(true);

  const recognitionRef = useRef<any>(null);

  const getLocale = (langCode: string): string => {
    const found = languages.find((l) => l.code === langCode);
    return found?.iso || langCode || 'en-US';
  };

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechSupported(false);
      recognitionRef.current = null;
      return;
    }
    setSpeechSupported(true);

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = getLocale(sourceLang);

    recognition.onresult = (event: any) => {
      let fullText = '';
      for (let i = 0; i < event.results.length; i++) {
        fullText += event.results[i][0].transcript;
      }
      setTranscript(fullText);
    };

    recognition.onerror = (event: any) => {
      if (event.error !== 'no-speech') {
        setError(`Microphone error: ${event.error}`);
        setIsRecording(false);
      }
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.stop();
      } catch {
        // Ignore cleanup stop errors
      }
    };
  }, [sourceLang, languages]);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      setError('Speech recognition is not supported in this browser. Type in the transcript box below.');
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
      if (transcript.trim()) {
        handleTranslate();
      }
    } else {
      setTranscript('');
      setResult(null);
      setError(null);
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (err: any) {
        setError(err.message || 'Could not start microphone');
      }
    }
  };

  const handleSwap = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
    if (result?.translated_text) {
      setTranscript(result.translated_text);
      setResult(null);
    }
  };

  const handleTranslate = async () => {
    if (!transcript.trim()) return;
    if (sourceLang === targetLang) {
      setError('Source and target languages must be different.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await translateText(transcript, sourceLang, targetLang, 2, false);
      setResult(res);
      speakText(res.translated_text, targetLang);
    } catch (err: any) {
      setError(err.message || 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  const speakText = (text: string, langCode: string) => {
    if (!('speechSynthesis' in window) || !text) return;
    if (isPlaying) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = getLocale(langCode);
    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);
    window.speechSynthesis.speak(utterance);
  };

  const handleCopy = () => {
    if (!result?.translated_text) return;
    navigator.clipboard.writeText(result.translated_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const isSameLanguage = sourceLang === targetLang;

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 animate-fade-up">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Real-Time Voice Translation
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Speak or type a transcript — translate with the local PyTorch Transformer and play audio.
        </p>
      </div>

      {/* Language selectors with Swap */}
      <div className="bg-white dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-sm grid grid-cols-1 sm:grid-cols-[1fr,auto,1fr] gap-4 items-center mb-6">
        <LanguageSelector
          label="Spoken Input Language"
          languages={languages}
          selectedCode={sourceLang}
          onSelect={(c) => {
            if (c === targetLang) setTargetLang(sourceLang);
            setSourceLang(c);
          }}
        />
        <div className="flex justify-center sm:pt-5">
          <button
            type="button"
            onClick={handleSwap}
            title="Swap languages"
            className="w-10 h-10 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:text-brand-600 active:rotate-180 transition-all flex items-center justify-center"
          >
            <ArrowLeftRight className="w-4 h-4" />
          </button>
        </div>
        <LanguageSelector
          label="Spoken Output Language"
          languages={languages}
          selectedCode={targetLang}
          onSelect={(c) => {
            if (c === sourceLang) setSourceLang(targetLang);
            setTargetLang(c);
          }}
        />
      </div>

      {/* Microphone Recording Center */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-8 text-center shadow-lg mb-6">
        {speechSupported ? (
          <div className="flex flex-col items-center justify-center gap-4 mb-6">
            <div className="relative">
              {isRecording && (
                <span className="absolute -inset-4 rounded-full bg-rose-500/25 animate-ping" />
              )}
              <button
                type="button"
                onClick={toggleRecording}
                className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-lg active:scale-95 ${
                  isRecording
                    ? 'bg-rose-600 text-white scale-105 shadow-rose-500/40'
                    : 'bg-gradient-to-tr from-brand-600 to-indigo-600 hover:from-brand-700 hover:to-indigo-700 text-white shadow-brand-500/30'
                }`}
              >
                {isRecording ? <MicOff className="w-10 h-10 animate-pulse" /> : <Mic className="w-10 h-10" />}
              </button>
            </div>

            <div>
              <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                {isRecording ? 'Listening... Tap to Stop & Translate' : 'Tap Microphone to Speak'}
              </span>
              <p className="text-xs text-slate-400 mt-1">
                Speech locale: {getLocale(sourceLang)} → {getLocale(targetLang)}
              </p>
            </div>
          </div>
        ) : (
          <div className="mb-6 p-3.5 rounded-2xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 text-xs text-amber-800 dark:text-amber-200 flex items-center justify-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>Browser Web Speech Recognition is unavailable here. Type text below to translate and hear speech playback.</span>
          </div>
        )}

        {/* Editable Transcript Area */}
        <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 text-left">
          <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">
            Speech Transcript (Editable)
          </label>
          <textarea
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            rows={3}
            placeholder="Spoken words will appear here in real time (or type manually)..."
            className="w-full bg-transparent text-base text-slate-800 dark:text-slate-100 placeholder:text-slate-400 resize-none focus:outline-none"
          />
        </div>

        {transcript.trim() && !isRecording && (
          <div className="mt-4 flex justify-end">
            <button
              type="button"
              onClick={handleTranslate}
              disabled={loading || isSameLanguage}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-700 hover:to-indigo-700 text-white text-xs font-bold shadow-md transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Translating...' : 'Translate & Speak'}</span>
            </button>
          </div>
        )}
      </div>

      {result && (
        <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-6 shadow-md animate-fade-up">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-3 border-b border-slate-100 dark:border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-wider text-brand-600 dark:text-brand-400">
                Spoken Translation ({result.target_language_name})
              </span>
              <span className="text-[11px] font-mono text-slate-400">({result.inference_time}s)</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCopy}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 text-xs font-semibold"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              {'speechSynthesis' in window && (
                <button
                  type="button"
                  onClick={() => speakText(result.translated_text, result.target_language || targetLang)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 text-xs font-semibold hover:bg-brand-100 transition-colors"
                >
                  <Volume2 className="w-4 h-4" />
                  <span>{isPlaying ? 'Stop Audio' : 'Play Audio'}</span>
                </button>
              )}
            </div>
          </div>
          <p className="text-xl font-medium text-slate-900 dark:text-white leading-relaxed">
            {result.translated_text}
          </p>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
