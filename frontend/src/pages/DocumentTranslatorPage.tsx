import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, Download, AlertCircle, Clock, Sparkles } from 'lucide-react';
import { Language } from '../types';
import { translateDocument } from '../services/api';
import { LanguageSelector } from '../components/LanguageSelector';

interface DocumentTranslatorPageProps {
  languages: Language[];
}

export const DocumentTranslatorPage: React.FC<DocumentTranslatorPageProps> = ({ languages }) => {
  const [file, setFile] = useState<File | null>(null);
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('hi');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    filename: string;
    total_paragraphs: number;
    translated_text: string;
    inference_time: number;
  } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      const ext = selected.name.split('.').pop()?.toLowerCase();
      if (ext !== 'txt' && ext !== 'pdf') {
        setError('Only .txt and .pdf files are supported.');
        return;
      }
      setFile(selected);
      setError(null);
      setResult(null);
    }
  };

  const handleTranslate = async () => {
    if (!file) return;
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

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
          Document Translation
        </h1>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Upload TXT or PDF documents. The Custom Multilingual Transformer parses paragraphs, translates in chunks, and preserves paragraph structure.
        </p>
      </div>

      {/* Language selectors */}
      <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <LanguageSelector
          label="Document Language"
          languages={languages}
          selectedCode={sourceLang}
          onSelect={setSourceLang}
        />
        <LanguageSelector
          label="Translate Into"
          languages={languages}
          selectedCode={targetLang}
          onSelect={setTargetLang}
        />
      </div>

      {/* Upload Zone */}
      <div className="bg-white dark:bg-slate-900 rounded-3xl border-2 border-dashed border-slate-300 dark:border-slate-700 p-8 text-center hover:border-brand-500 transition-colors shadow-sm mb-6">
        <input
          type="file"
          id="doc-upload"
          accept=".txt,.pdf"
          onChange={handleFileChange}
          className="hidden"
        />
        <label htmlFor="doc-upload" className="cursor-pointer flex flex-col items-center gap-3">
          <div className="w-16 h-16 rounded-2xl bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 flex items-center justify-center">
            <UploadCloud className="w-8 h-8" />
          </div>
          <div>
            <span className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Click to upload or drag & drop document
            </span>
            <p className="text-xs text-slate-400 mt-1">
              Supports .TXT and .PDF formats (Max 5MB)
            </p>
          </div>
        </label>

        {file && (
          <div className="mt-6 inline-flex items-center gap-3 px-4 py-2.5 rounded-xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs">
            <FileText className="w-4 h-4 text-brand-500" />
            <span className="font-semibold text-slate-800 dark:text-slate-200">{file.name}</span>
            <span className="text-slate-400">({(file.size / 1024).toFixed(1)} KB)</span>
            <button
              onClick={() => {
                setFile(null);
                setResult(null);
              }}
              className="text-rose-500 hover:text-rose-600 font-bold ml-2"
            >
              ✕
            </button>
          </div>
        )}

        {file && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleTranslate}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-brand-600 hover:bg-brand-700 text-white text-sm font-semibold shadow-md shadow-brand-500/25 transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Processing & Translating Document...' : 'Translate Document'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Progress or Loading State */}
      {loading && (
        <div className="p-6 rounded-2xl bg-brand-50 dark:bg-brand-950/40 border border-brand-200 dark:border-brand-900 text-center mb-6">
          <div className="w-8 h-8 rounded-full border-3 border-brand-600 border-t-transparent animate-spin mx-auto mb-2"></div>
          <span className="text-sm font-semibold text-brand-700 dark:text-brand-300">
            Extracting text & running neural machine translation...
          </span>
          <p className="text-xs text-brand-600/80 dark:text-brand-400/80 mt-1">
            Segmenting text into paragraphs for Transformer generation
          </p>
        </div>
      )}

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs mb-6">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Result Card */}
      {result && (
        <div className="bg-white dark:bg-slate-900 rounded-3xl border border-slate-200 dark:border-slate-800 shadow-lg p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 dark:border-slate-800 pb-4 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                <h3 className="font-bold text-slate-800 dark:text-slate-100">
                  Translation Complete
                </h3>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                <span>{result.total_paragraphs} paragraphs translated</span>
                <span>•</span>
                <span className="flex items-center gap-1 font-mono">
                  <Clock className="w-3 h-3" /> {result.inference_time}s
                </span>
              </div>
            </div>

            <button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-md shadow-emerald-600/20 transition-colors"
            >
              <Download className="w-4 h-4" />
              <span>Download Translated Text</span>
            </button>
          </div>

          {/* Document Preview */}
          <div className="max-h-96 overflow-y-auto p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 font-sans text-sm text-slate-800 dark:text-slate-200 whitespace-pre-wrap leading-relaxed">
            {result.translated_text}
          </div>
        </div>
      )}
    </div>
  );
};
