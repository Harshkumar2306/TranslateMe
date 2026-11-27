import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check, Globe, X, Sparkles } from 'lucide-react';
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
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  return (
    <div className="relative w-full" ref={dropdownRef}>
      <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1.5 uppercase tracking-wider">
        {label}
      </label>

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
            {displayNative ? (
              <span className="text-xs text-slate-400 dark:text-slate-500 truncate block">
                {displayNative} • {selectedLang?.script}
              </span>
            ) : selectedCode === 'auto' ? (
              <span className="text-xs text-slate-400 dark:text-slate-500 truncate block">
                Unicode script detection
              </span>
            ) : null}
          </div>
        </div>
        <ChevronDown className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-50 sm:hidden animate-in fade-in duration-200"
            onClick={() => setIsOpen(false)}
          />

          <div className="fixed inset-x-0 bottom-0 z-50 rounded-t-3xl sm:rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-2xl overflow-hidden backdrop-blur-xl sm:absolute sm:inset-x-auto sm:bottom-auto sm:top-full sm:mt-2 sm:w-full sm:min-w-[260px] sm:left-0 max-h-[85vh] flex flex-col animate-fade-up">
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
                aria-label="Close language selector"
                className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-2 space-y-1 text-sm overflow-y-auto">
              {allowAuto && (
                <button
                  type="button"
                  onClick={() => {
                    onSelect('auto');
                    setIsOpen(false);
                  }}
                  className={`w-full min-h-[44px] flex items-center justify-between px-3.5 py-2.5 rounded-xl text-left transition-colors ${
                    selectedCode === 'auto'
                      ? 'bg-brand-50 text-brand-700 dark:bg-brand-950/70 dark:text-brand-300 font-semibold border border-brand-200/60 dark:border-brand-800/60'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Sparkles className="w-4 h-4 text-brand-500 flex-shrink-0" />
                    <div>
                      <div className="font-semibold text-slate-900 dark:text-white">Auto Detect</div>
                      <div className="text-[11px] text-slate-400">Detect script automatically</div>
                    </div>
                  </div>
                  {selectedCode === 'auto' && <Check className="w-4 h-4 text-brand-600 dark:text-brand-400 flex-shrink-0" />}
                </button>
              )}

              {languages.map((lang) => {
                const isSelected = selectedCode === lang.code;
                return (
                  <button
                    key={lang.code}
                    type="button"
                    onClick={() => {
                      onSelect(lang.code);
                      setIsOpen(false);
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
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
