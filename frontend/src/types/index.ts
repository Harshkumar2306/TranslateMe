export interface Language {
  code: string;
  name: string;
  native_name: string;
  script: string;
  iso?: string;
}

export interface TranslationResult {
  original_text: string;
  translated_text: string;
  source_language: string;
  target_language: string;
  source_language_name?: string;
  target_language_name?: string;
  detected_language?: string | null;
  inference_time: number;
  model_name: string;
  alternatives: string[];
}

export interface HistoryItem {
  id: number;
  original_text: string;
  translated_text: string;
  source_language: string;
  source_language_name?: string;
  target_language: string;
  target_language_name?: string;
  detected_language?: string | null;
  inference_time: number;
  model_name: string;
  created_at: string;
}

export interface ModelInfo {
  model_name: string;
  model_type: string;
  num_parameters: string;
  supported_languages_count: number;
  framework: string;
  device: string;
  license: string;
  attribution: string;
  limitations: string[];
}

export interface AnalyticsData {
  total_translations: number;
  average_inference_time_ms: number;
  top_source_languages: { language: string; count: number }[];
  top_target_languages: { language: string; count: number }[];
  recent_activity: HistoryItem[];
}

export interface LabInfo {
  custom_transformer: {
    name: string;
    encoder_layers: number;
    decoder_layers: number;
    attention_heads: number;
    d_model: number;
    d_ff: number;
    training_history?: {
      epochs: number[];
      loss: number[];
      lr: number[];
    };
  };
  evaluation_benchmark?: {
    timestamp: string;
    benchmark_dataset: string;
    models: {
      nllb_pretrained?: {
        model_name: string;
        bleu: number;
        chrf: number;
        avg_latency_sec: number;
        throughput_chars_per_sec: number;
        samples: { src: string; ref: string; hyp: string }[];
      };
      custom_transformer?: {
        model_name: string;
        bleu: number;
        chrf: number;
        avg_latency_sec: number;
        throughput_chars_per_sec: number;
        samples: { src: string; ref: string; hyp: string }[];
      };
    };
  };
  transformer_components: {
    name: string;
    formula: string;
    description: string;
  }[];
}
