import React, { useState } from 'react';
import {
  Languages,
  Mic,
  FileText,
  History,
  Cpu,
  Info,
  BarChart2,
  Sun,
  Moon,
  Sparkles,
  MoreHorizontal,
  X,
  ChevronRight,
} from 'lucide-react';

import { ModelInfo } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isDark: boolean;
  toggleTheme: () => void;
  modelInfo?: ModelInfo | null;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isDark,
  toggleTheme,
  modelInfo,
}) => {
  const [isMoreMenuOpen, setIsMoreMenuOpen] = useState(false);

  const mainNavItems = [
    { id: 'translator', label: 'Translator', icon: Languages },
    { id: 'voice', label: 'Voice', icon: Mic },
    { id: 'document', label: 'Documents', icon: FileText },
    { id: 'history', label: 'History', icon: History },
    { id: 'lab', label: 'DL Lab', fullLabel: 'Deep Learning Lab', icon: Cpu },
    { id: 'analytics', label: 'Analytics', fullLabel: 'Analytics', icon: BarChart2 },
    { id: 'model', label: 'Specs', fullLabel: 'Model Specs', icon: Info },
  ];

  const mobilePrimaryTabs = [
    { id: 'translator', label: 'Translate', icon: Languages },
    { id: 'voice', label: 'Voice', icon: Mic },
    { id: 'document', label: 'Docs', icon: FileText },
    { id: 'history', label: 'History', icon: History },
  ];

  const mobileMoreTabs = [
    {
      id: 'lab',
      label: 'Deep Learning Lab',
      desc: 'Attention maps, loss curves & Transformer metrics',
      icon: Cpu,
    },
    {
      id: 'analytics',
      label: 'Analytics Dashboard',
      desc: 'Latency & language usage computed from SQLite',
      icon: BarChart2,
    },
    {
      id: 'model',
      label: 'Model Specifications',
      desc: 'Active PyTorch Transformer architecture & weights',
      icon: Info,
    },
  ];

  const isMoreActive = ['lab', 'analytics', 'model'].includes(activeTab);

  return (
    <>
      <header className="sticky top-0 z-40 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800/80 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-3 h-16">
            <div
              onClick={() => setActiveTab('translator')}
              className="flex items-center gap-2.5 sm:gap-3 cursor-pointer group select-none shrink-0"
            >
              <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-brand-600 via-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-brand-500/25 group-hover:scale-105 active:scale-95 transition-transform shrink-0">
                <Sparkles className="w-5 h-5" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 whitespace-nowrap">
                  <span className="font-extrabold text-lg sm:text-xl tracking-tight bg-gradient-to-r from-slate-900 via-slate-800 to-slate-700 dark:from-white dark:via-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                    TranslateMe
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 hidden 2xl:block whitespace-nowrap">
                  From-Scratch PyTorch NMT
                </p>
              </div>
            </div>

            <nav className="hidden lg:flex items-center gap-0.5 p-1 rounded-2xl bg-slate-100/85 dark:bg-slate-800/70 border border-slate-200/70 dark:border-slate-700/60 shrink-0">
              {mainNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-1.5 px-2.5 xl:px-3 py-1.5 rounded-xl text-xs xl:text-sm font-semibold whitespace-nowrap shrink-0 transition-all select-none ${
                      isActive
                        ? 'bg-white dark:bg-slate-900 text-brand-600 dark:text-brand-300 shadow-sm'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-white/60 dark:hover:bg-slate-800'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="hidden 2xl:inline">{item.fullLabel || item.label}</span>
                    <span className="2xl:hidden">{item.label}</span>
                  </button>
                );
              })}
            </nav>

            <div className="flex items-center gap-2.5 shrink-0">
              {modelInfo && (
                <button
                  type="button"
                  onClick={() => setActiveTab('model')}
                  title="View active PyTorch model specifications"
                  className="hidden xl:inline-flex items-center gap-2 px-3 py-1.5 rounded-xl border border-emerald-200/80 dark:border-emerald-900/60 bg-emerald-50/70 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 text-xs font-semibold hover:bg-emerald-100/80 dark:hover:bg-emerald-950/70 transition-colors whitespace-nowrap"
                >
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="font-mono">
                    {modelInfo.num_parameters.split('(')[1]?.replace(')', '') || modelInfo.num_parameters}
                  </span>
                  <span className="text-emerald-500/60">•</span>
                  <span className="uppercase font-mono text-[10px]">{modelInfo.device}</span>
                </button>
              )}
              <button
                type="button"
                onClick={toggleTheme}
                aria-label="Toggle theme"
                className="w-10 h-10 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-800/80 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 active:scale-95 transition-all shrink-0"
              >
                {isDark ? (
                  <Sun className="w-4 h-4 text-amber-400" />
                ) : (
                  <Moon className="w-4 h-4 text-slate-700" />
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="hidden sm:flex lg:hidden overflow-x-auto py-2 px-4 gap-1.5 border-t border-slate-100 dark:border-slate-800/80 no-scrollbar">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-brand-600 text-white shadow-xs'
                    : 'text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      <div className="sm:hidden fixed bottom-0 inset-x-0 z-40 bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border-t border-slate-200/80 dark:border-slate-800/80 pb-[max(8px,env(safe-area-inset-bottom))] shadow-lg">
        <div className="grid grid-cols-5 items-center h-14 px-2">
          {mobilePrimaryTabs.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setActiveTab(item.id);
                  setIsMoreMenuOpen(false);
                }}
                className={`flex flex-col items-center justify-center py-1 rounded-xl transition-all active:scale-90 select-none ${
                  isActive
                    ? 'text-brand-600 dark:text-brand-400 font-bold'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <div className={`p-1 rounded-lg transition-colors ${isActive ? 'bg-brand-50 dark:bg-brand-950/80' : ''}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <span className="text-[10px] mt-0.5 tracking-tight">{item.label}</span>
              </button>
            );
          })}

          <button
            type="button"
            onClick={() => setIsMoreMenuOpen(!isMoreMenuOpen)}
            className={`flex flex-col items-center justify-center py-1 rounded-xl transition-all active:scale-90 select-none ${
              isMoreActive || isMoreMenuOpen
                ? 'text-brand-600 dark:text-brand-400 font-bold'
                : 'text-slate-500 dark:text-slate-400'
            }`}
          >
            <div className={`p-1 rounded-lg transition-colors ${isMoreActive || isMoreMenuOpen ? 'bg-brand-50 dark:bg-brand-950/80' : ''}`}>
              <MoreHorizontal className="w-4 h-4" />
            </div>
            <span className="text-[10px] mt-0.5 tracking-tight">More</span>
          </button>
        </div>
      </div>

      {isMoreMenuOpen && (
        <div className="sm:hidden fixed inset-0 z-50 animate-in fade-in duration-200">
          <div
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm"
            onClick={() => setIsMoreMenuOpen(false)}
          />
          <div className="fixed inset-x-0 bottom-0 z-50 rounded-t-3xl border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 pb-[max(24px,env(safe-area-inset-bottom))] shadow-2xl animate-in slide-in-from-bottom duration-250">
            <div className="w-12 h-1 bg-slate-300 dark:bg-slate-700 rounded-full mx-auto mb-4" />
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="font-bold text-base text-slate-900 dark:text-white">
                  Research & Insights
                </h3>
                <p className="text-xs text-slate-500">
                  Deep Learning architecture & benchmarks
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsMoreMenuOpen(false)}
                className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2">
              {mobileMoreTabs.map((item) => {
                const Icon = item.icon;
                const isItemActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(item.id);
                      setIsMoreMenuOpen(false);
                    }}
                    className={`w-full flex items-center justify-between p-3.5 rounded-2xl text-left border transition-all active:scale-[0.98] ${
                      isItemActive
                        ? 'border-brand-500 bg-brand-50/60 dark:bg-brand-950/40 text-brand-700 dark:text-brand-300'
                        : 'border-slate-100 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-800/40 text-slate-800 dark:text-slate-200'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center justify-center text-brand-600 dark:text-brand-400">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="font-bold text-sm block">{item.label}</span>
                        <p className="text-xs text-slate-400 dark:text-slate-500 line-clamp-1">
                          {item.desc}
                        </p>
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </>
  );
};
