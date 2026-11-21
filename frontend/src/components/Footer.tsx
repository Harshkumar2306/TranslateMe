import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-auto border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/50 py-8 text-xs text-slate-500 dark:text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-800 dark:text-slate-200 text-sm">LinguaAI</span>
            <span>—</span>
            <span>Universal Multilingual Neural Machine Translation System</span>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span className="font-medium">Local PyTorch Transformer Engine</span>
            </div>
            <div className="flex items-center gap-1 text-slate-400">
              <Cpu className="w-3.5 h-3.5" />
              <span>NLLB-200 + Custom Transformer</span>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-100 dark:border-slate-800/80 flex flex-col sm:flex-row justify-between items-center gap-2">
          <p>
            Academic Deep Learning Research Project. Model weights: Meta AI NLLB-200 (CC-BY-NC 4.0 License).
          </p>
          <div className="flex items-center gap-2 text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Zero External API Calls • 100% Local Inference</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
