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
  parameters?: string;
  layers?: number;
  encoder_layers?: number;
  decoder_layers?: number;
  attention_heads?: number;
  head_dim?: number;
  d_model?: number;
  d_ff?: number;
  max_len?: number;
  dropout?: number;
  vocab_size?: string;
  tokenizer_type?: string;
  supported_languages_count: number;
  framework: string;
  device: string;
  license: string;
  attribution: string;
  limitations: string[];
  is_modern?: boolean;
  weight_tying?: string;
  use_rope?: boolean;
  use_qk_norm?: boolean;
  use_layerscale?: boolean;
  soft_cap?: number | null;
  checkpoint_size_mb?: number;
  weight_dtype?: string;
  epochs_completed?: number | null;
  global_step?: number | null;
  best_val_loss?: number | null;
  training_details?: {
    epochs_completed?: number | null;
    global_step?: number | null;
    best_val_loss?: number | null;
    checkpoint_size_mb?: number;
    weight_dtype?: string;
    train_history_points?: number;
    val_history_points?: number;
    soft_cap?: number | null;
    kv_cache_decoding?: string;
  };
  references?: { title: string; citation: string }[];
}

export interface AnalyticsData {
  total_translations: number;
  average_inference_time_ms: number;
  unique_languages_count: number;
  active_directions_count: number;
  top_source_languages: { language: string; count: number }[];
  top_target_languages: { language: string; count: number }[];
  top_language_pairs: { pair: string; count: number }[];
  recent_activity: HistoryItem[];
}

export interface DocumentTranslateResult {
  filename: string;
  total_paragraphs: number;
  translated_paragraphs: number;
  truncated: boolean;
  translated_text: string;
  inference_time: number;
  source_language: string;
  target_language: string;
}

export interface AttentionResult {
  source_text: string;
  translated_text: string;
  source_language: string;
  target_language: string;
  source_tokens: string[];
  target_tokens: string[];
  raw_source_tokens?: string[];
  raw_target_tokens?: string[];
  attention_matrix: number[][];
  decoder_layer?: number;
  num_heads?: number;
}

export interface TrainingCurvePoint {
  step: number;
  train_loss?: number;
  val_loss?: number;
}

export interface LabInfo {
  custom_transformer: {
    name: string;
    is_modern: boolean;
    encoder_layers: number;
    decoder_layers: number;
    attention_heads: number;
    head_dim: number;
    d_model: number;
    d_ff: number;
    parameters: number;
    shared_vocab_size: number;
    use_rope: boolean;
    use_qk_norm: boolean;
    use_layerscale: boolean;
    soft_cap?: number | null;
    share_embeddings: boolean;
    epochs_completed?: number | null;
    global_step?: number | null;
    best_val_loss?: number | null;
    checkpoint_size_mb?: number | null;
    weight_dtype?: string | null;
    device: string;
    training_curve: TrainingCurvePoint[];
    training_x_label: string;
    training_summary?: {
      start_train_loss?: number | null;
      end_train_loss?: number | null;
      start_val_loss?: number | null;
      end_val_loss?: number | null;
      best_val_loss?: number | null;
      loss_reduction_pct?: number | null;
      train_points_count?: number;
      val_points_count?: number;
    };
  };
  evaluation_benchmark?: Record<
    string,
    {
      direction: string;
      src_lang: string;
      tgt_lang: string;
      test_sentence_count: number;
      bleu: number;
      chrf: number;
      avg_latency_sec: number;
      throughput_chars_per_sec: number;
      empty_rate?: number;
      repetition_rate?: number;
      decoding_algorithm?: string;
      samples?: { source: string; reference: string; hypothesis: string }[];
    }
  >;
  evaluation_summary?: {
    directions_count: number;
    test_sentence_count_per_dir: number;
    avg_bleu: number;
    avg_chrf: number;
    avg_latency_sec: number;
    decoding_algorithm: string;
    dataset_source: string;
  } | null;
  transformer_components?: {
    name: string;
    formula: string;
    description: string;
  }[];
}
