import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Volume2, Sparkles, AlertCircle, ArrowRight, Play, Square } from 'lucide-react';
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

  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // Initialize Web Speech Recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;

      // Map language code to BCP-47 locale
      const currentSource = languages.find(l => l.code === sourceLang);
      recognition.lang = currentSource?.iso || 'en-US';

      recognition.onresult = (event: any) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        if (event.error !== 'no-speech') {
          setError(`Microphone error: ${event.error}`);
          setIsRecording(false);
        }
      };

      recognition.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
    }
  }, [sourceLang, languages]);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      setError('Speech recognition is not supported in this browser. Please use Chrome/Edge or enter text manually.');
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

  const handleTranslate = async () => {
    if (!transcript.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await translateText(transcript, sourceLang, targetLang, 2, false);
      setResult(res);
      // Auto speak translated text
      speakText(res.translated_text);
    } catch (err: any) {
      setError(err.message || 'Translation failed');
    } finally {
      setLoading(false);
    }
  };

  const speakText = (text: string) => {
    if (!('speechSynthesis' in window) || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setIsPlaying(true);
    utterance.onend = () => setIsPlaying(false);
    utterance.onerror = () => setIsPlaying(false);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Real-Time Voice Translation
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Speak in your native language, translate via local Transformer, and hear immediate spoken output.
        </p>
      </div>

      {/* Language selectors */}
      <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <LanguageSelector
          label="Spoken Input Language"
          languages={languages}
          selectedCode={sourceLang}
          onSelect={setSourceLang}
        />
        <LanguageSelector
          label="Spoken Output Language"
          languages={languages}
          selectedCode={targetLang}
          onSelect={setTargetLang}
        />
      </div>

      {/* Microphone Recording Center */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 p-8 text-center shadow-lg mb-6">
        <div className="flex flex-col items-center justify-center gap-4">
          <div className="relative">
            {isRecording && (
              <span className="absolute -inset-4 rounded-full bg-rose-500/20 animate-ping"></span>
            )}
            <button
              onClick={toggleRecording}
              className={`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-lg ${
                isRecording
                  ? 'bg-rose-600 text-white scale-110 shadow-rose-500/40'
                  : 'bg-brand-600 hover:bg-brand-700 text-white shadow-brand-500/30'
              }`}
            >
              {isRecording ? <MicOff className="w-10 h-10 animate-pulse" /> : <Mic className="w-10 h-10" />}
            </button>
          </div>

          <div>
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              {isRecording ? 'Listening... Click to Finish' : 'Click Microphone to Start Speaking'}
            </span>
            <p className="text-xs text-slate-400 mt-1">
              Supports browser speech recognition and local audio synthesis
            </p>
          </div>
        </div>

        {/* Live Transcript Area */}
        <div className="mt-8 p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700 text-left min-h-[90px]">
          <span className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
            Speech Recognition Transcript
          </span>
          {transcript ? (
            <p className="text-base text-slate-800 dark:text-slate-100 font-medium">
              {transcript}
            </p>
          ) : (
            <p className="text-sm text-slate-400 italic">
              Spoken words will appear here in real time...
            </p>
          )}
        </div>

        {/* Translate action button */}
        {transcript && !isRecording && (
          <div className="mt-4 flex justify-end">
            <button
              onClick={handleTranslate}
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold shadow-md transition-all"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Translating...' : 'Translate & Speak'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Translation Output Card */}
      {result && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-md animate-in fade-in slide-in-from-bottom-2 duration-200">
          <div className="flex items-center justify-between mb-3 border-b border-slate-100 dark:border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-brand-600 dark:text-brand-400">
                Spoken Translation ({result.target_language_name})
              </span>
              <span className="text-[11px] text-slate-400">({result.inference_time}s)</span>
            </div>
            <button
              onClick={() => speakText(result.translated_text)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 text-xs font-medium hover:bg-brand-100 transition-colors"
            >
              <Volume2 className="w-4 h-4" />
              <span>{isPlaying ? 'Playing Audio...' : 'Play Again'}</span>
            </button>
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
