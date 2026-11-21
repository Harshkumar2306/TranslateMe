import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, Check, Globe } from 'lucide-react';
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
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedLang = languages.find((l) => l.code === selectedCode);
  const displayName = selectedCode === 'auto' ? 'Auto Detect' : (selectedLang ? selectedLang.name : selectedCode);
  const displayNative = selectedLang?.native_name;

  const filteredLanguages = languages.filter(
    (l) =>
      l.name.toLowerCase().includes(search.toLowerCase()) ||
      l.native_name.toLowerCase().includes(search.toLowerCase()) ||
      l.script.toLowerCase().includes(search.toLowerCase()) ||
      l.code.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="relative" ref={dropdownRef}>
      <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5 uppercase tracking-wider">
        {label}
      </label>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between gap-2 px-3.5 py-2.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/90 text-left hover:border-brand-500 dark:hover:border-brand-500 transition-all shadow-sm focus:outline-none focus:ring-2 focus:ring-brand-500/20"
      >
        <div className="flex items-center gap-2 truncate">
          <Globe className="w-4 h-4 text-brand-500 flex-shrink-0" />
          <span className="font-medium text-slate-800 dark:text-slate-200 truncate">{displayName}</span>
          {displayNative && (
            <span className="text-xs text-slate-400 dark:text-slate-500 hidden sm:inline">
              ({displayNative})
            </span>
          )}
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-2 w-72 sm:w-80 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 shadow-xl overflow-hidden backdrop-blur-lg animate-in fade-in zoom-in-95 duration-100">
          {/* Search box */}
          <div className="p-3 border-b border-slate-100 dark:border-slate-700/60">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search language or script..."
                autoFocus
                className="w-full pl-9 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-brand-500/30"
              />
            </div>
          </div>

          {/* Language list */}
          <div className="max-h-64 overflow-y-auto p-1 text-sm">
            {allowAuto && (
              <button
                type="button"
                onClick={() => {
                  onSelect('auto');
                  setIsOpen(false);
                  setSearch('');
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-colors ${
                  selectedCode === 'auto'
                    ? 'bg-brand-50 text-brand-700 dark:bg-brand-950/60 dark:text-brand-300 font-semibold'
                    : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-brand-500"></span>
                  <span>Auto Detect</span>
                </div>
                {selectedCode === 'auto' && <Check className="w-4 h-4 text-brand-600 dark:text-brand-400" />}
              </button>
            )}

            {filteredLanguages.length === 0 ? (
              <div className="p-4 text-center text-xs text-slate-400">No supported languages match</div>
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
                    className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-left transition-colors ${
                      isSelected
                        ? 'bg-brand-50 text-brand-700 dark:bg-brand-950/60 dark:text-brand-300 font-semibold'
                        : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700/50'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span>{lang.name}</span>
                        <span className="text-xs text-slate-400 dark:text-slate-500 font-normal">
                          {lang.native_name}
                        </span>
                      </div>
                      <div className="text-[10px] text-slate-400 dark:text-slate-500">
                        {lang.script} script • <span className="font-mono">{lang.code}</span>
                      </div>
                    </div>
                    {isSelected && <Check className="w-4 h-4 text-brand-600 dark:text-brand-400 flex-shrink-0" />}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
};
