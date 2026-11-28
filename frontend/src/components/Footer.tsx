import React from 'react';
import { Cpu } from 'lucide-react';
import { Language, ModelInfo } from '../types';

interface FooterProps {
  modelInfo?: ModelInfo | null;
  languages?: Language[];
}

export const Footer: React.FC<FooterProps> = ({ modelInfo, languages = [] }) => {
  const langCodes = languages.length > 0 ? languages.map((l) => l.code.toUpperCase()).join(' • ') : 'Multilingual';
  const langCount = languages.length || modelInfo?.supported_languages_count || 0;

  return (
    <footer className="mt-auto border-t border-slate-200/80 dark:border-slate-800/80 bg-white/70 dark:bg-slate-900/50 backdrop-blur-md py-6 mb-16 sm:mb-0 text-xs text-slate-500 dark:text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-800 dark:text-slate-200 text-sm">TranslateMe</span>
            <span>—</span>
            <span>
              {langCount > 0 ? `${langCount}-Language Neural Machine Translation (${langCodes})` : 'Neural Machine Translation'}
            </span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4">
            {modelInfo && (
              <>
                <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="font-medium">Local PyTorch Engine ({modelInfo.device.toUpperCase()})</span>
                </div>
                <div className="flex items-center gap-1 text-slate-500 dark:text-slate-400">
                  <Cpu className="w-3.5 h-3.5 text-brand-500" />
                  <span>{modelInfo.model_name}</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </footer>
  );
};
