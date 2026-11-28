import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { TranslatorPage } from './pages/TranslatorPage';
import { VoiceTranslatorPage } from './pages/VoiceTranslatorPage';
import { DocumentTranslatorPage } from './pages/DocumentTranslatorPage';
import { HistoryPage } from './pages/HistoryPage';
import { DeepLearningLabPage } from './pages/DeepLearningLabPage';
import { ModelInfoPage } from './pages/ModelInfoPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { Language, ModelInfo } from './types';
import { fetchLanguages, fetchModelInfo } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState('translator');
  const [languages, setLanguages] = useState<Language[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('translateme_theme') ?? localStorage.getItem('lingua_theme');
    return (
      saved === 'dark' ||
      (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)
    );
  });

  const [reuseData, setReuseData] = useState<{
    text: string;
    src: string;
    tgt: string;
  } | null>(null);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('translateme_theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('translateme_theme', 'light');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [activeTab]);

  useEffect(() => {
    fetchLanguages()
      .then(setLanguages)
      .catch((err) => console.error('Failed to load languages:', err));
    fetchModelInfo()
      .then(setModelInfo)
      .catch((err) => console.error('Failed to load model specs:', err));
  }, []);

  const handleReuseTranslation = (text: string, srcLang: string, tgtLang: string) => {
    setReuseData({ text, src: srcLang, tgt: tgtLang });
    setActiveTab('translator');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 studio-mesh text-slate-900 dark:text-slate-100 transition-colors duration-200">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDark={isDark}
        toggleTheme={toggleTheme}
        modelInfo={modelInfo}
      />

      <main className="flex-1 pb-28 sm:pb-16">
        {activeTab === 'translator' && (
          <TranslatorPage
            languages={languages}
            initialSourceText={reuseData?.text || ''}
            initialSourceLang={reuseData?.src || 'en'}
            initialTargetLang={reuseData?.tgt || 'hi'}
            onReuseConsumed={() => setReuseData(null)}
          />
        )}

        {activeTab === 'voice' && <VoiceTranslatorPage languages={languages} />}

        {activeTab === 'document' && <DocumentTranslatorPage languages={languages} />}

        {activeTab === 'history' && (
          <HistoryPage languages={languages} onReuseTranslation={handleReuseTranslation} />
        )}

        {activeTab === 'lab' && <DeepLearningLabPage modelInfo={modelInfo} languages={languages} />}

        {activeTab === 'analytics' && <AnalyticsPage />}

        {activeTab === 'model' && <ModelInfoPage modelInfo={modelInfo} />}
      </main>

      <Footer modelInfo={modelInfo} languages={languages} />
    </div>
  );
}

export default App;
