import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Clock, Globe, ArrowRight, Activity, RefreshCw, AlertCircle } from 'lucide-react';
import { AnalyticsData } from '../types';
import { fetchAnalytics } from '../services/api';

const COLORS = ['#4f6ef7', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];

export const AnalyticsPage: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadAnalytics = () => {
    setLoading(true);
    setError(null);
    fetchAnalytics()
      .then(setData)
      .catch((err) => setError(err.message || 'Failed to load analytics'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 animate-fade-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
            Translation Analytics Dashboard
          </h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            Live usage metrics, active directions, and inference latency computed from SQLite history.
          </p>
        </div>

        <button
          type="button"
          onClick={loadAnalytics}
          disabled={loading}
          className="self-start sm:self-auto flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 text-xs font-semibold hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 flex items-center gap-2 text-rose-600 dark:text-rose-400 text-xs">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5 mb-8">
        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-10 h-10 rounded-xl bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 flex items-center justify-center mb-3">
            <Activity className="w-5 h-5" />
          </div>
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
            Total Translations
          </span>
          <span className="text-2xl font-bold text-slate-900 dark:text-white mt-1 block">
            {loading && !data ? '—' : (data?.total_translations ?? 0)}
          </span>
          <span className="text-[11px] text-slate-400">Recorded sessions in SQLite</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-3">
            <Clock className="w-5 h-5" />
          </div>
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
            Average Latency
          </span>
          <span className="text-2xl font-bold text-slate-900 dark:text-white mt-1 block">
            {loading && !data ? '—' : `${data?.average_inference_time_ms ?? 0} ms`}
          </span>
          <span className="text-[11px] text-slate-400">Per translation request</span>
        </div>

        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center mb-3">
            <Globe className="w-5 h-5" />
          </div>
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
            Active Directions
          </span>
          <span className="text-2xl font-bold text-slate-900 dark:text-white mt-1 block">
            {loading && !data ? '—' : `${data?.active_directions_count ?? 0}`}
          </span>
          <span className="text-[11px] text-slate-400">
            Across {data?.unique_languages_count ?? 0} unique languages used
          </span>
        </div>
      </div>

      {/* Visual Analytics Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <h2 className="text-base font-bold text-slate-900 dark:text-white mb-1">
            Most Popular Target Languages
          </h2>
          <p className="text-xs text-slate-400 mb-4">Frequency of destination languages</p>

          <div className="h-64">
            {data?.top_target_languages && data.top_target_languages.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.top_target_languages}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="language" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} domain={[0, (dataMax: number) => Math.max(4, Math.ceil(dataMax * 1.15))]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '12px', color: '#fff', fontSize: '12px' }}
                  />
                  <Bar dataKey="count" fill="#4f6ef7" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No translations recorded yet
              </div>
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <h2 className="text-base font-bold text-slate-900 dark:text-white mb-1">
            Source Language Distribution
          </h2>
          <p className="text-xs text-slate-400 mb-4">Origin language proportions</p>

          <div className="h-64">
            {data?.top_source_languages && data.top_source_languages.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={data.top_source_languages}
                    cx="50%"
                    cy="48%"
                    innerRadius={50}
                    outerRadius={74}
                    paddingAngle={4}
                    dataKey="count"
                    nameKey="language"
                  >
                    {data.top_source_languages.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '12px', color: '#fff', fontSize: '12px' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    height={28}
                    wrapperStyle={{ fontSize: '11px' }}
                    formatter={(value: string) => {
                      const entry = data.top_source_languages.find((item) => item.language === value);
                      return entry ? `${value} (${entry.count})` : value;
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-slate-400">
                No translations recorded yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Top Language Pairs */}
      {data?.top_language_pairs && data.top_language_pairs.length > 0 && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm mb-8">
          <h2 className="text-base font-bold text-slate-900 dark:text-white mb-4">
            Translation Direction Breakdown
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {data.top_language_pairs.map((item) => (
              <div
                key={item.pair}
                className="flex items-center justify-between p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 text-xs"
              >
                <span className="font-mono font-semibold text-slate-800 dark:text-slate-200">{item.pair}</span>
                <span className="px-2 py-0.5 rounded-md bg-brand-50 dark:bg-brand-950/60 text-brand-700 dark:text-brand-300 font-bold">
                  {item.count}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activity Table */}
      {data?.recent_activity && data.recent_activity.length > 0 && (
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <h2 className="text-base font-bold text-slate-900 dark:text-white mb-4">
            Recent Translation Logs
          </h2>
          <div className="space-y-3">
            {data.recent_activity.map((item) => (
              <div
                key={item.id}
                className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 text-xs"
              >
                <div className="flex-1 truncate">
                  <div className="flex items-center gap-2 font-medium text-slate-800 dark:text-slate-200 mb-1">
                    <span className="text-brand-600 dark:text-brand-400 font-semibold">{item.source_language_name || item.source_language}</span>
                    <ArrowRight className="w-3 h-3 text-slate-400" />
                    <span className="text-brand-600 dark:text-brand-400 font-semibold">{item.target_language_name || item.target_language}</span>
                  </div>
                  <p className="text-slate-500 truncate">{item.original_text}</p>
                </div>
                <div className="text-right flex sm:flex-col items-center sm:items-end justify-between font-mono text-[11px] text-slate-400">
                  <span>{item.inference_time}s</span>
                  <span>{new Date(item.created_at).toLocaleTimeString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
