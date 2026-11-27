import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, Download, AlertCircle, Clock, Sparkles, ArrowLeftRight, Copy, Check } from 'lucide-react';
import { Language, DocumentTranslateResult } from '../types';
import { translateDocument } from '../services/api';
import { LanguageSelector } from '../components/LanguageSelector';

interface DocumentTranslatorPageProps {
  languages: Language[];
}

export const DocumentTranslatorPage: React.FC<DocumentTranslatorPageProps> = ({ languages }) => {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [result, setResult] = useState<DocumentTranslateResult | null>(null);

  const acceptFile = (selected: File) => {
    const ext = selected.name.split('.').pop()?.toLowerCase();
    if (ext !== 'txt' && ext !== 'pdf') {
      setError('Only .txt and .pdf files are supported.');
      return;
    }
    if (selected.size > 10 * 1024 * 1024) {
      setError('File exceeds the 10 MB upload limit.');
      return;
    }
    setFile(selected);
    setError(null);
    setResult(null);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      acceptFile(e.target.files[0]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      acceptFile(e.dataTransfer.files[0]);
    }
  };

  const handleSwap = () => {
    setSourceLang(targetLang);
    setTargetLang(sourceLang);
  };

  const handleTranslate = async () => {
    if (!file) return;
    if (sourceLang === targetLang) {
      setError('Source and target languages must be different.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await translateDocument(file, sourceLang, targetLang);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Document translation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result) return;
    const blob = new Blob([result.translated_text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `translated_${result.filename.replace(/\.[^/.]+$/, '')}.txt`;
    a.click();
    URL.revokeObjectURL(url);
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
          Document Translation
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Upload TXT or PDF documents. Our local Transformer parses paragraphs and preserves structure.
        </p>
      </div>

      {/* Language selectors with Swap */}
      <div className="bg-white dark:bg-slate-900 p-5 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-sm grid grid-cols-1 sm:grid-cols-[1fr,auto,1fr] gap-4 items-center mb-6">
        <LanguageSelector
          label="Document Language"
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
          label="Translate Into"
          languages={languages}
          selectedCode={targetLang}
          onSelect={(c) => {
            if (c === sourceLang) setSourceLang(targetLang);
            setTargetLang(c);
          }}
        />
      </div>

      {/* Drag-and-Drop Upload Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragEnter={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`bg-white dark:bg-slate-900 rounded-3xl border-2 border-dashed p-8 text-center transition-all shadow-sm mb-6 ${
          isDragging
            ? 'border-brand-500 bg-brand-50/40 dark:bg-brand-950/30 scale-[1.01]'
            : 'border-slate-300 dark:border-slate-700 hover:border-brand-500'
        }`}
      >
        <input
          type="file"
          id="doc-upload"
          accept=".txt,.pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <label htmlFor="doc-upload" className="cursor-pointer flex flex-col items-center gap-3">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors ${
            isDragging
              ? 'bg-brand-600 text-white'
              : 'bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400'
          }`}>
            <UploadCloud className="w-8 h-8" />
          </div>
          <div>
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              {isDragging ? 'Drop your document here' : 'Click to upload or drag & drop document'}
            </span>
            <p className="text-xs text-slate-400 mt-1">
              Supports .TXT and .PDF formats (Max 10 MB, up to 50 paragraphs per request)
            </p>
          </div>
        </label>

        {file && (
          <div className="mt-6 inline-flex items-center gap-3 px-4 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs">
            <FileText className="w-4 h-4 text-brand-500" />
            <span className="font-semibold text-slate-800 dark:text-slate-200">{file.name}</span>
            <span className="text-slate-400">({(file.size / 1024).toFixed(1)} KB)</span>
            <button
              type="button"
              onClick={() => {
                setFile(null);
                setResult(null);
              }}
              aria-label="Remove file"
              className="text-rose-500 hover:text-rose-600 font-bold ml-2"
            >
              ✕
            </button>
          </div>
        )}

        {file && (
          <div className="mt-6 flex justify-center">
            <button
              type="button"
              onClick={handleTranslate}
              disabled={loading || isSameLanguage}
              className="flex items-center gap-2 px-7 py-3 rounded-2xl bg-gradient-to-r from-brand-600 to-indigo-600 hover:from-brand-700 hover:to-indigo-700 text-white text-sm font-bold shadow-lg shadow-brand-500/25 transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Processing & Translating Document...' : 'Translate Document'}</span>
            </button>
          </div>
        )}
      </div>

      {loading && (
        <div className="p-6 rounded-2xl bg-brand-50 dark:bg-brand-950/40 border border-brand-200 dark:border-brand-900 text-center mb-6">
          <div className="w-8 h-8 rounded-full border-2 border-brand-600 border-t-transparent animate-spin mx-auto mb-2" />
          <span className="text-sm font-semibold text-brand-700 dark:text-brand-300">
            Extracting text & running local Transformer translation...
          </span>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs mb-6">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {result && (
        <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-lg p-6 animate-fade-up">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 dark:border-slate-800 pb-4 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                <h3 className="font-bold text-slate-800 dark:text-slate-100">
                  Translation Complete
                </h3>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                <span>
                  {result.truncated
                    ? `${result.translated_paragraphs} of ${result.total_paragraphs} paragraphs translated`
                    : `${result.translated_paragraphs} paragraphs translated`}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3 h-3" /> {result.inference_time}s
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 text-xs font-semibold hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
              <button
                type="button"
                onClick={handleDownload}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-md shadow-emerald-600/20 transition-colors"
              >
                <Download className="w-4 h-4" />
                <span>Download .TXT</span>
              </button>
            </div>
          </div>

          <div className="max-h-96 overflow-y-auto p-4 rounded-2xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 font-sans text-sm text-slate-800 dark:text-slate-200 whitespace-pre-wrap leading-relaxed">
            {result.translated_text}
          </div>
        </div>
      )}
    </div>
  );
};
