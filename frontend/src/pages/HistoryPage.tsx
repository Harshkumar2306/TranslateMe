import React, { useState, useEffect } from 'react';
import { Search, Trash2, ArrowUpRight, Copy, Check, Clock, Calendar, ArrowRight, Volume2, RefreshCw, AlertCircle } from 'lucide-react';
import { HistoryItem, Language } from '../types';
import { fetchHistory, deleteHistoryItem, clearAllHistory } from '../services/api';

interface HistoryPageProps {
  languages?: Language[];
  onReuseTranslation: (text: string, srcLang: string, tgtLang: string) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ languages = [], onReuseTranslation }) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [langFilter, setLangFilter] = useState<string>('all');
  const [copiedId, setCopiedId] = useState<number | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const items = await fetchHistory(100);
      setHistory(items);
    } catch (err: any) {
      setError(err.message || 'Failed to load translation history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleDelete = async (id: number) => {
    try {
      await deleteHistoryItem(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
    } catch (err: any) {
      setError(err.message || 'Failed to delete entry');
    }
  };

  const handleClearAll = async () => {
    try {
      await clearAllHistory();
      setHistory([]);
      setConfirmClear(false);
    } catch (err: any) {
      setError(err.message || 'Failed to clear history');
    }
  };

  const handleCopy = (id: number, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleSpeak = (text: string, langCode: string) => {
    if (!('speechSynthesis' in window) || !text) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const found = languages.find((l) => l.code === langCode);
    u.lang = found?.iso || langCode || 'en-US';
    window.speechSynthesis.speak(u);
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.original_text.toLowerCase().includes(search.toLowerCase()) ||
      item.translated_text.toLowerCase().includes(search.toLowerCase()) ||
      (item.source_language_name && item.source_language_name.toLowerCase().includes(search.toLowerCase())) ||
      (item.target_language_name && item.target_language_name.toLowerCase().includes(search.toLowerCase()));

    const matchesLang =
      langFilter === 'all' ||
      item.source_language === langFilter ||
      item.target_language === langFilter;

    return matchesSearch && matchesLang;
  });

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 animate-fade-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
            Translation History
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Stored locally in SQLite. Search, filter, reuse, or manage past translations.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            type="button"
            onClick={loadHistory}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 text-xs font-semibold hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          {history.length > 0 && (
            confirmClear ? (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleClearAll}
                  className="px-3 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-sm transition-colors"
                >
                  Confirm Delete All
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmClear(false)}
                  className="px-3 py-2 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 text-xs font-semibold"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-rose-200 dark:border-rose-900/60 text-rose-600 dark:text-rose-400 text-xs font-semibold hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear All ({history.length})</span>
              </button>
            )
          )}
        </div>
      </div>

      {error && (
        <div className="mb-5 p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Search + Language Filter Pills */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search keywords, source, or translated text..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/25 shadow-sm"
          />
        </div>
        {languages.length > 0 && (
          <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar">
            <button
              type="button"
              onClick={() => setLangFilter('all')}
              className={`px-3 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-colors ${
                langFilter === 'all'
                  ? 'bg-brand-600 text-white'
                  : 'bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400'
              }`}
            >
              All
            </button>
            {languages.map((l) => (
              <button
                key={l.code}
                type="button"
                onClick={() => setLangFilter(l.code)}
                className={`px-3 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-colors ${
                  langFilter === l.code
                    ? 'bg-brand-600 text-white'
                    : 'bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 hover:border-brand-400'
                }`}
              >
                {l.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-32 rounded-2xl bg-slate-200/60 dark:bg-slate-800/50 animate-pulse" />
          ))}
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="p-12 text-center rounded-2xl border border-dashed border-slate-200 dark:border-slate-800 text-slate-400 text-sm">
          {search || langFilter !== 'all'
            ? 'No history entries matched your current search or language filter.'
            : 'No translations saved yet. Start translating on the Translator tab!'}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item) => (
            <div
              key={item.id}
              className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm hover:border-brand-500/50 transition-colors"
            >
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800/80 pb-3 mb-3 text-xs text-slate-400">
                <div className="flex items-center gap-2 font-medium text-slate-700 dark:text-slate-300">
                  <span className="font-semibold text-brand-600 dark:text-brand-400">
                    {item.source_language_name || item.source_language}
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-400" />
                  <span className="font-semibold text-violet-600 dark:text-violet-400">
                    {item.target_language_name || item.target_language}
                  </span>
                  {item.detected_language && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-300 font-semibold">
                      Auto-detected
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1 font-mono text-[11px]">
                    <Clock className="w-3 h-3" /> {item.inference_time}s
                  </span>
                  <span className="flex items-center gap-1 text-[11px]">
                    <Calendar className="w-3 h-3" /> {new Date(item.created_at).toLocaleString()}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Original
                  </span>
                  <p className="text-slate-800 dark:text-slate-200 leading-relaxed font-sans">
                    {item.original_text}
                  </p>
                </div>
                <div>
                  <span className="text-[11px] font-semibold text-brand-600 dark:text-brand-400 uppercase tracking-wider block mb-1">
                    Translation
                  </span>
                  <p className="text-slate-900 dark:text-white font-medium leading-relaxed font-sans">
                    {item.translated_text}
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs">
                {'speechSynthesis' in window && (
                  <button
                    type="button"
                    onClick={() => handleSpeak(item.translated_text, item.target_language)}
                    className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    <span>Listen</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={() => handleCopy(item.id, item.translated_text)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors"
                >
                  {copiedId === item.id ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedId === item.id ? 'Copied' : 'Copy'}</span>
                </button>

                <button
                  type="button"
                  onClick={() => onReuseTranslation(item.original_text, item.source_language, item.target_language)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 hover:bg-brand-100 transition-colors font-semibold"
                >
                  <ArrowUpRight className="w-3.5 h-3.5" />
                  <span>Reuse</span>
                </button>

                <button
                  type="button"
                  onClick={() => handleDelete(item.id)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
                  title="Delete entry"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
