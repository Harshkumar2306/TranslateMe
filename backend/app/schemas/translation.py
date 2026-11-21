"""
Pydantic Schemas for TranslateMe API
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Input text to translate")
    source_language: str = Field(..., description="Source language code (e.g., 'eng_Latn' or 'auto')")
    target_language: str = Field(..., description="Target language code (e.g., 'hin_Deva')")
    num_beams: Optional[int] = Field(2, ge=1, le=5, description="Number of beams for beam search")
    return_alternatives: Optional[bool] = Field(False, description="Whether to return alternative translations")
    model_type: Optional[str] = Field("nllb", description="'nllb' or 'custom'")


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

    class Config:
        from_attributes = True


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


class AnalyticsResponse(BaseModel):
    total_translations: int
    average_inference_time_ms: float
    top_source_languages: List[dict]
    top_target_languages: List[dict]
    recent_activity: List[HistoryItem]


class DocumentTranslateResponse(BaseModel):
    filename: str
    total_paragraphs: int
    translated_text: str
    inference_time: float
    source_language: str
    target_language: str
