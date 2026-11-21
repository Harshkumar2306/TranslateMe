import React, { useState, useEffect } from 'react';
import { Search, Trash2, ArrowUpRight, Copy, Check, Clock, Calendar, ArrowRight, Layers } from 'lucide-react';
import { HistoryItem } from '../types';
import { fetchHistory, deleteHistoryItem, clearAllHistory } from '../services/api';

interface HistoryPageProps {
  onReuseTranslation: (text: string, srcLang: string, tgtLang: string) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ onReuseTranslation }) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const items = await fetchHistory(100);
      setHistory(items);
    } catch (err) {
      console.error(err);
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
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to delete all translation history?')) return;
    try {
      await clearAllHistory();
      setHistory([]);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCopy = (id: number, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const filteredHistory = history.filter(
    (item) =>
      item.original_text.toLowerCase().includes(search.toLowerCase()) ||
      item.translated_text.toLowerCase().includes(search.toLowerCase()) ||
      (item.source_language_name && item.source_language_name.toLowerCase().includes(search.toLowerCase())) ||
      (item.target_language_name && item.target_language_name.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
            Translation History
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Stored locally in SQLite database. View, search, reuse, and manage previous translations.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={handleClearAll}
            className="self-start sm:self-auto flex items-center gap-1.5 px-3 py-2 rounded-xl border border-rose-200 dark:border-rose-900/60 text-rose-600 dark:text-rose-400 text-xs font-semibold hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear All History</span>
          </button>
        )}
      </div>

      {/* Search Bar */}
      <div className="relative mb-6">
        <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by keywords, language, or translated text..."
          className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/25 shadow-sm"
        />
      </div>

      {/* History Items List */}
      {loading ? (
        <div className="p-12 text-center text-slate-400 text-sm">
          Loading history records...
        </div>
      ) : filteredHistory.length === 0 ? (
        <div className="p-12 text-center rounded-2xl border border-dashed border-slate-200 dark:border-slate-800 text-slate-400 text-sm">
          {search ? 'No history entries matched your search query.' : 'No translations saved yet. Start translating on the homepage!'}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredHistory.map((item) => (
            <div
              key={item.id}
              className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm hover:border-brand-500/50 transition-colors"
            >
              {/* Header meta */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800/80 pb-3 mb-3 text-xs text-slate-400">
                <div className="flex items-center gap-2 font-medium text-slate-700 dark:text-slate-300">
                  <span className="font-semibold text-brand-600 dark:text-brand-400">
                    {item.source_language_name || item.source_language}
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-400" />
                  <span className="font-semibold text-brand-600 dark:text-brand-400">
                    {item.target_language_name || item.target_language}
                  </span>
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

              {/* Text Body */}
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

              {/* Actions */}
              <div className="flex items-center justify-end gap-2 mt-4 pt-3 border-t border-slate-100 dark:border-slate-800/60 text-xs">
                <button
                  onClick={() => handleCopy(item.id, item.translated_text)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-300 transition-colors"
                >
                  {copiedId === item.id ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedId === item.id ? 'Copied' : 'Copy'}</span>
                </button>

                <button
                  onClick={() => onReuseTranslation(item.original_text, item.source_language, item.target_language)}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 hover:bg-brand-100 transition-colors font-medium"
                >
                  <ArrowUpRight className="w-3.5 h-3.5" />
                  <span>Reuse in Translator</span>
                </button>

                <button
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
