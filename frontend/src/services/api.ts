import { Language, TranslationResult, HistoryItem, ModelInfo, AnalyticsData, LabInfo } from '../types';

const API_BASE = '/api';

export async function fetchLanguages(): Promise<Language[]> {
  const res = await fetch(`${API_BASE}/languages`);
  if (!res.ok) throw new Error('Failed to load supported languages');
  return res.json();
}

export async function translateText(
  text: string,
  sourceLanguage: string,
  targetLanguage: string,
  numBeams: number = 2,
  returnAlternatives: boolean = false
): Promise<TranslationResult> {
  const res = await fetch(`${API_BASE}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      source_language: sourceLanguage,
      target_language: targetLanguage,
      num_beams: numBeams,
      return_alternatives: returnAlternatives
    }),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Translation failed' }));
    throw new Error(error.detail || 'Translation failed');
  }
  return res.json();
}

export async function detectLanguage(text: string): Promise<{ detected_code: string; detected_name: string; confidence: number }> {
  const res = await fetch(`${API_BASE}/detect-language`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error('Language detection failed');
  return res.json();
}

export async function fetchHistory(limit = 50): Promise<HistoryItem[]> {
  const res = await fetch(`${API_BASE}/history?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch translation history');
  return res.json();
}

export async function deleteHistoryItem(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/history/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete history item');
}

export async function clearAllHistory(): Promise<void> {
  const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear history');
}

export async function translateDocument(
  file: File,
  sourceLanguage: string,
  targetLanguage: string
): Promise<{
  filename: string;
  total_paragraphs: number;
  translated_text: string;
  inference_time: number;
}> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('source_language', sourceLanguage);
  formData.append('target_language', targetLanguage);

  const res = await fetch(`${API_BASE}/translate/document`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Document translation failed' }));
    throw new Error(error.detail || 'Document translation failed');
  }
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error('Failed to fetch model information');
  return res.json();
}

export async function fetchAnalytics(): Promise<AnalyticsData> {
  const res = await fetch(`${API_BASE}/analytics`);
  if (!res.ok) throw new Error('Failed to fetch analytics');
  return res.json();
}

export async function fetchLabInfo(): Promise<LabInfo> {
  const res = await fetch(`${API_BASE}/lab/info`);
  if (!res.ok) throw new Error('Failed to fetch Deep Learning Lab data');
  return res.json();
}
