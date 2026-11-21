import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { TranslatorPage } from './pages/TranslatorPage';
import { VoiceTranslatorPage } from './pages/VoiceTranslatorPage';
import { DocumentTranslatorPage } from './pages/DocumentTranslatorPage';
import { HistoryPage } from './pages/HistoryPage';
import { DeepLearningLabPage } from './pages/DeepLearningLabPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { Language } from './types';
import { fetchLanguages } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState('translator');
  const [languages, setLanguages] = useState<Language[]>([]);
  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('lingua_theme') === 'dark' ||
      (!('lingua_theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });

  // Prefill state for reusing from history
  const [reuseData, setReuseData] = useState<{
    text: string;
    src: string;
    tgt: string;
  } | null>(null);

  // Apply dark mode class to html element
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('lingua_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('lingua_theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);

  // Load languages
  useEffect(() => {
    fetchLanguages()
      .then(setLanguages)
      .catch((err) => console.error('Failed to load languages:', err));
  }, []);

  const handleReuseTranslation = (text: string, srcLang: string, tgtLang: string) => {
    setReuseData({ text, src: srcLang, tgt: tgtLang });
    setActiveTab('translator');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDark={isDark}
        toggleTheme={toggleTheme}
      />

      <main className="flex-1 pb-28 sm:pb-16">
        {activeTab === 'translator' && (
          <TranslatorPage
            languages={languages}
            initialSourceText={reuseData?.text || ''}
            initialSourceLang={reuseData?.src || 'en'}
            initialTargetLang={reuseData?.tgt || 'hi'}
          />
        )}

        {activeTab === 'voice' && <VoiceTranslatorPage languages={languages} />}

        {activeTab === 'document' && <DocumentTranslatorPage languages={languages} />}

        {activeTab === 'history' && (
          <HistoryPage onReuseTranslation={handleReuseTranslation} />
        )}

        {activeTab === 'lab' && <DeepLearningLabPage />}

        {activeTab === 'analytics' && <AnalyticsPage />}

        {activeTab === 'model' && <ModelInfoPage />}
      </main>

      <Footer />
    </div>
  );
}

export default App;
