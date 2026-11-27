"""
Pydantic Schemas for TranslateMe API
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict, field_serializer


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to translate")
    source_language: str = Field(..., description="Source language code ('en', 'hi', 'ta', 'te', or 'auto')")
    target_language: str = Field(..., description="Target language code ('en', 'hi', 'ta', 'te')")
    num_beams: Optional[int] = Field(2, ge=1, le=5, description="Number of beams for beam search")
    return_alternatives: Optional[bool] = Field(False, description="Whether to return alternative translations")


class TranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    detected_language: Optional[str] = None
    inference_time: float
    model_name: str
    source_language_name: Optional[str] = None
    target_language_name: Optional[str] = None
    alternatives: List[str] = []


class AttentionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Input text for cross-attention extraction")
    source_lang: str = Field("en", description="Source language code ('en', 'hi', 'ta', 'te', or 'auto')")
    target_lang: str = Field("hi", description="Target language code ('en', 'hi', 'ta', 'te')")


class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str
    script: str
    iso: Optional[str] = None


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class DetectLanguageResponse(BaseModel):
    detected_code: Optional[str]
    detected_name: Optional[str]
    confidence: float


class HistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    original_text: str
    translated_text: str
    source_language: str
    source_language_name: Optional[str] = None
    target_language: str
    target_language_name: Optional[str] = None
    detected_language: Optional[str] = None
    inference_time: float
    model_name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    num_parameters: str
    supported_languages_count: int
    framework: str
    device: str
    license: str
    attribution: str
    limitations: List[str]
    parameters: Optional[str] = None
    layers: Optional[int] = None
    encoder_layers: Optional[int] = None
    decoder_layers: Optional[int] = None
    attention_heads: Optional[int] = None
    head_dim: Optional[int] = None
    d_model: Optional[int] = None
    d_ff: Optional[int] = None
    max_len: Optional[int] = None
    dropout: Optional[float] = None
    vocab_size: Optional[str] = None
    tokenizer_type: Optional[str] = None
    is_modern: bool = False
    weight_tying: Optional[str] = None
    use_rope: bool = False
    use_qk_norm: bool = False
    use_layerscale: bool = False
    soft_cap: Optional[float] = None
    checkpoint_size_mb: Optional[float] = None
    weight_dtype: Optional[str] = None
    epochs_completed: Optional[int] = None
    global_step: Optional[int] = None
    best_val_loss: Optional[float] = None
    training_details: Optional[Dict[str, Any]] = None
    references: List[Dict[str, str]] = []


class AnalyticsResponse(BaseModel):
    total_translations: int
    average_inference_time_ms: float
    unique_languages_count: int = 0
    active_directions_count: int = 0
    top_source_languages: List[dict]
    top_target_languages: List[dict]
    top_language_pairs: List[dict] = []
    recent_activity: List[HistoryItem]


class DocumentTranslateResponse(BaseModel):
    filename: str
    total_paragraphs: int
    translated_paragraphs: int
    truncated: bool = False
    translated_text: str
    inference_time: float
    source_language: str
    target_language: str
