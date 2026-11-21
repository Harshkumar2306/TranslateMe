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
  ChevronRight
} from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isDark: boolean;
  toggleTheme: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isDark,
  toggleTheme,
}) => {
  const [isMoreMenuOpen, setIsMoreMenuOpen] = useState(false);

  const mainNavItems = [
    { id: 'translator', label: 'Translator', icon: Languages },
    { id: 'voice', label: 'Voice', icon: Mic },
    { id: 'document', label: 'Documents', icon: FileText },
    { id: 'history', label: 'History', icon: History },
    { id: 'lab', label: 'DL Lab', fullLabel: 'Deep Learning Lab', icon: Cpu, badge: 'Research' },
    { id: 'analytics', label: 'Analytics', fullLabel: 'Analytics', icon: BarChart2 },
    { id: 'model', label: 'Specs', fullLabel: 'Model Specs', icon: Info },
  ];

  // Primary 4 items for bottom bar on phones
  const mobilePrimaryTabs = [
    { id: 'translator', label: 'Translate', icon: Languages },
    { id: 'voice', label: 'Voice', icon: Mic },
    { id: 'document', label: 'Docs', icon: FileText },
    { id: 'history', label: 'History', icon: History },
  ];

  // Secondary items for the "More" bottom sheet on phones
  const mobileMoreTabs = [
    {
      id: 'lab',
      label: 'Deep Learning Lab',
      desc: 'Attention maps, loss curves & Transformer metrics',
      icon: Cpu,
      badge: 'Research'
    },
    {
      id: 'analytics',
      label: 'Analytics Dashboard',
      desc: 'Throughput, latency distribution & language breakdown',
      icon: BarChart2
    },
    {
      id: 'model',
      label: 'Model Specifications',
      desc: 'NLLB-200 architecture & Meta AI research attribution',
      icon: Info
    },
  ];

  const isMoreActive = ['lab', 'analytics', 'model'].includes(activeTab);

  return (
    <>
      {/* Top Header Bar */}
      <header className="sticky top-0 z-40 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200/80 dark:border-slate-800/80 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div
              onClick={() => setActiveTab('translator')}
              className="flex items-center gap-3 cursor-pointer group select-none"
            >
              <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-gradient-to-tr from-brand-600 via-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-brand-500/25 group-hover:scale-105 active:scale-95 transition-transform">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-extrabold text-lg sm:text-xl tracking-tight bg-gradient-to-r from-slate-900 via-slate-800 to-slate-700 dark:from-white dark:via-slate-100 dark:to-slate-300 bg-clip-text text-transparent">
                    TranslateMe
                  </span>
                  <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-brand-50 text-brand-700 dark:bg-brand-950/90 dark:text-brand-300 border border-brand-200/80 dark:border-brand-800">
                    NLLB-200
                  </span>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:block">
                  Universal Multilingual Transformer Translation
                </p>
              </div>
            </div>

            {/* Desktop & Tablet Navigation */}
            <nav className="hidden md:flex items-center gap-1">
              {mainNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold transition-all select-none ${
                      isActive
                        ? 'bg-brand-50 text-brand-700 dark:bg-brand-950/70 dark:text-brand-300 shadow-xs'
                        : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-slate-800/60'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{item.fullLabel || item.label}</span>
                    {item.badge && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-indigo-500/90 text-white font-bold tracking-wide">
                        {item.badge}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>

            {/* Right Quick Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={toggleTheme}
                aria-label="Toggle theme"
                className="w-10 h-10 rounded-xl border border-slate-200 dark:border-slate-800 flex items-center justify-center text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 active:scale-95 transition-all"
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

        {/* Tablet Horizontal Scroll Navigation (Medium screens) */}
        <div className="hidden sm:flex md:hidden overflow-x-auto py-2 px-4 gap-1.5 border-t border-slate-100 dark:border-slate-800/80 no-scrollbar">
          {mainNavItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
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

      {/* Mobile Native App Bottom Navigation Bar (Phones) */}
      <div className="sm:hidden fixed bottom-0 inset-x-0 z-40 bg-white/90 dark:bg-slate-900/90 backdrop-blur-xl border-t border-slate-200/80 dark:border-slate-800/80 pb-[max(8px,env(safe-area-inset-bottom))] shadow-lg">
        <div className="grid grid-cols-5 items-center h-14 px-2">
          {mobilePrimaryTabs.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
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

          {/* 5th Tab: More / Research Menu */}
          <button
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

      {/* Mobile "More" Bottom Sheet Modal */}
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
                  Explore Deep Learning architectures & benchmarks
                </p>
              </div>
              <button
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
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-sm">{item.label}</span>
                          {item.badge && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-indigo-500 text-white font-semibold">
                              {item.badge}
                            </span>
                          )}
                        </div>
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
