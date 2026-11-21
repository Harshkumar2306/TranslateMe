import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, Check, Globe, X, Sparkles } from 'lucide-react';
import { Language } from '../types';

interface LanguageSelectorProps {
  languages: Language[];
  selectedCode: string;
  onSelect: (code: string) => void;
  allowAuto?: boolean;
  label: string;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  languages,
  selectedCode,
  onSelect,
  allowAuto = false,
  label,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'indian' | 'global'>('all');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Close when clicking outside on desktop
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // Focus search input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => searchInputRef.current?.focus(), 50);
    } else {
      setSearch('');
    }
  }, [isOpen]);

  // Lock body scroll on mobile modal
  useEffect(() => {
    if (isOpen && window.innerWidth < 640) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const selectedLang = languages.find((l) => l.code === selectedCode);
  const displayName = selectedCode === 'auto' ? 'Auto Detect' : (selectedLang ? selectedLang.name : selectedCode);
  const displayNative = selectedLang?.native_name;

  // Categorization helpers
  const indianScripts = ['Devanagari', 'Tamil', 'Telugu', 'Malayalam', 'Bengali', 'Gujarati', 'Kannada', 'Gurmukhi', 'Odia'];

  const filteredLanguages = languages.filter((l) => {
    const matchesSearch =
      l.name.toLowerCase().includes(search.toLowerCase()) ||
      l.native_name.toLowerCase().includes(search.toLowerCase()) ||
      l.script.toLowerCase().includes(search.toLowerCase()) ||
      l.code.toLowerCase().includes(search.toLowerCase());

    if (!matchesSearch) return false;

    if (selectedCategory === 'indian') {
      return indianScripts.some(s => l.script.toLowerCase().includes(s.toLowerCase()));
    }
    if (selectedCategory === 'global') {
      return !indianScripts.some(s => l.script.toLowerCase().includes(s.toLowerCase()));
    }
    return true;
  });

  return (
    <div className="relative w-full" ref={dropdownRef}>
      <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5 uppercase tracking-wider">
        {label}
      </label>

      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full min-h-[48px] flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-2xl border border-slate-200 dark:border-slate-700/80 bg-white dark:bg-slate-800/90 hover:border-brand-500 dark:hover:border-brand-500 transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-500/20 active:scale-[0.99]"
      >
        <div className="flex items-center gap-2.5 truncate min-w-0">
          <div className="w-7 h-7 rounded-lg bg-brand-50 dark:bg-brand-950/60 text-brand-600 dark:text-brand-400 flex items-center justify-center flex-shrink-0">
            {selectedCode === 'auto' ? <Sparkles className="w-3.5 h-3.5" /> : <Globe className="w-3.5 h-3.5" />}
          </div>
          <div className="truncate text-left">
            <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm truncate block">
              {displayName}
            </span>
            {displayNative && (
              <span className="text-xs text-slate-400 dark:text-slate-500 truncate block">
                {displayNative} • {selectedLang?.script}
              </span>
            )}
          </div>
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu (Responsive: Bottom Sheet on Mobile, Popover on Desktop) */}
      {isOpen && (
        <>
          {/* Mobile backdrop */}
          <div
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-50 sm:hidden animate-in fade-in duration-200"
            onClick={() => setIsOpen(false)}
          />

          <div
            className="fixed inset-x-0 bottom-0 z-50 rounded-t-3xl sm:rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-2xl overflow-hidden backdrop-blur-xl sm:absolute sm:inset-x-auto sm:bottom-auto sm:top-full sm:mt-2 sm:w-96 sm:right-0 max-h-[85vh] sm:max-h-[460px] flex flex-col animate-in slide-in-from-bottom-6 sm:slide-in-from-top-2 duration-200"
          >
            {/* Mobile Drag Indicator & Header */}
            <div className="p-3.5 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <div className="sm:hidden w-12 h-1 bg-slate-300 dark:bg-slate-700 rounded-full mx-auto absolute top-2 inset-x-0" />
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-brand-500" />
                <span className="font-bold text-sm text-slate-900 dark:text-white">
                  Select {label}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Search Box */}
            <div className="p-3 border-b border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-850">
              <div className="relative">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                <input
                  ref={searchInputRef}
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search language, script, or code (e.g. Hindi, Deva)..."
                  className="w-full pl-9 pr-8 py-2 text-sm rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => setSearch('')}
                    className="absolute right-2.5 top-2.5 p-0.5 rounded-md text-slate-400 hover:text-slate-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Filter Tabs */}
              <div className="flex items-center gap-1.5 mt-2.5 overflow-x-auto no-scrollbar">
                <button
                  type="button"
                  onClick={() => setSelectedCategory('all')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                    selectedCategory === 'all'
                      ? 'bg-brand-600 text-white'
                      : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
                  }`}
                >
                  All ({languages.length})
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedCategory('indian')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                    selectedCategory === 'indian'
                      ? 'bg-brand-600 text-white'
                      : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
                  }`}
                >
                  Indian Languages
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedCategory('global')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                    selectedCategory === 'global'
                      ? 'bg-brand-600 text-white'
                      : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-700'
                  }`}
                >
                  Global Languages
                </button>
              </div>
            </div>

            {/* Language List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1 text-sm divide-y divide-slate-100/50 dark:divide-slate-800/40">
              {allowAuto && !search && selectedCategory === 'all' && (
                <button
                  type="button"
                  onClick={() => {
                    onSelect('auto');
                    setIsOpen(false);
                    setSearch('');
                  }}
                  className={`w-full min-h-[44px] flex items-center justify-between px-3 py-2.5 rounded-xl text-left transition-colors ${
                    selectedCode === 'auto'
                      ? 'bg-brand-50 text-brand-700 dark:bg-brand-950/70 dark:text-brand-300 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-brand-500 animate-pulse"></span>
                    <div>
                      <div className="font-semibold text-slate-900 dark:text-white">Auto Detect</div>
                      <div className="text-[11px] text-slate-400">Statistical + script-based detection</div>
                    </div>
                  </div>
                  {selectedCode === 'auto' && <Check className="w-4 h-4 text-brand-600 dark:text-brand-400 flex-shrink-0" />}
                </button>
              )}

              {filteredLanguages.length === 0 ? (
                <div className="py-8 text-center text-sm text-slate-400">
                  No languages match "{search}"
                </div>
              ) : (
                filteredLanguages.map((lang) => {
                  const isSelected = selectedCode === lang.code;
                  return (
                    <button
                      key={lang.code}
                      type="button"
                      onClick={() => {
                        onSelect(lang.code);
                        setIsOpen(false);
                        setSearch('');
                      }}
                      className={`w-full min-h-[48px] flex items-center justify-between px-3.5 py-2.5 rounded-xl text-left transition-colors active:bg-slate-200 dark:active:bg-slate-700 ${
                        isSelected
                          ? 'bg-brand-50/80 text-brand-700 dark:bg-brand-950/70 dark:text-brand-300 font-semibold border border-brand-200/60 dark:border-brand-800/60'
                          : 'text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800/60'
                      }`}
                    >
                      <div className="min-w-0 pr-2">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-slate-900 dark:text-slate-100 truncate">
                            {lang.name}
                          </span>
                          <span className="text-xs text-slate-400 dark:text-slate-500 font-normal truncate">
                            {lang.native_name}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-medium bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
                            {lang.script}
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            {lang.code}
                          </span>
                        </div>
                      </div>
                      {isSelected && (
                        <Check className="w-4 h-4 text-brand-600 dark:text-brand-400 flex-shrink-0" />
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
