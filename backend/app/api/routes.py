"""
FastAPI REST API Routes for LinguaAI
"""

import os
import json
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.database.session import get_db, Base, engine
from backend.app.models.history import TranslationHistory
from backend.app.schemas.translation import (
    TranslateRequest,
    TranslateResponse,
    LanguageInfo,
    DetectLanguageRequest,
    DetectLanguageResponse,
    HistoryItem,
    ModelInfoResponse,
    AnalyticsResponse,
    DocumentTranslateResponse
)
from backend.app.services.nllb_service import translation_engine
from backend.app.services.language_codes import get_supported_languages, is_valid_language, get_language_name
from backend.app.services.language_detector import detect_language
from backend.app.services.document_service import extract_text_from_file, translate_document
from backend.app.core.device import get_device_info

# Create DB tables
Base.metadata.create_all(bind=engine)

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint displaying system and device status."""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "service": "LinguaAI Multilingual Translation API",
        "model_loaded": translation_engine.is_loaded,
        "device": device_info
    }


@router.get("/languages", response_model=List[LanguageInfo])
def list_languages():
    """Returns the list of all supported languages with metadata."""
    return get_supported_languages()


@router.post("/detect-language", response_model=DetectLanguageResponse)
def detect_lang(req: DetectLanguageRequest):
    """Detects the language of the provided text."""
    code, name, conf = detect_language(req.text)
    return DetectLanguageResponse(
        detected_code=code,
        detected_name=name,
        confidence=conf
    )


@router.post("/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest, db: Session = Depends(get_db)):
    """Translates text between any supported language pair using NLLB-200."""
    try:
        result = translation_engine.translate(
            text=req.text,
            source_lang=req.source_language,
            target_lang=req.target_language,
            num_beams=req.num_beams or 2,
            return_alternatives=req.return_alternatives or False
        )

        # Record in SQLite history
        history_entry = TranslationHistory(
            source_text=result["original_text"],
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            source_language_name=result["source_language_name"],
            target_language=result["target_language"],
            target_language_name=result["target_language_name"],
            detected_language=result["detected_language"],
            inference_time=result["inference_time"],
            model_name=result["model_name"]
        )
        db.add(history_entry)
        db.commit()

        return TranslateResponse(**result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/history", response_model=List[HistoryItem])
def get_translation_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Fetches user translation history ordered by most recent."""
    items = db.query(TranslationHistory).order_by(desc(TranslationHistory.created_at)).offset(offset).limit(limit).all()
    return [
        HistoryItem(
            id=item.id,
            original_text=item.source_text,
            translated_text=item.translated_text,
            source_language=item.source_language,
            source_language_name=item.source_language_name,
            target_language=item.target_language,
            target_language_name=item.target_language_name,
            detected_language=item.detected_language,
            inference_time=item.inference_time,
            model_name=item.model_name,
            created_at=item.created_at
        ) for item in items
    ]


@router.delete("/history/{item_id}")
def delete_history_item(item_id: int, db: Session = Depends(get_db)):
    """Deletes a specific history record."""
    item = db.query(TranslationHistory).filter(TranslationHistory.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="History entry not found")
    db.delete(item)
    db.commit()
    return {"message": "Entry successfully deleted", "id": item_id}


@router.delete("/history")
def clear_all_history(db: Session = Depends(get_db)):
    """Clears all translation history."""
    db.query(TranslationHistory).delete()
    db.commit()
    return {"message": "All history cleared"}


@router.post("/translate/document", response_model=DocumentTranslateResponse)
async def translate_doc(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    """Translates uploaded .txt or .pdf documents."""
    content = await file.read()
    paragraphs = extract_text_from_file(file, content)
    translated_doc, elapsed = translate_document(paragraphs, source_language, target_language)

    return DocumentTranslateResponse(
        filename=file.filename or "uploaded_document",
        total_paragraphs=len(paragraphs),
        translated_text=translated_doc,
        inference_time=elapsed,
        source_language=source_language,
        target_language=target_language
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_information():
    """Returns official model attribution, architecture, and limitations."""
    device_info = get_device_info()
    return ModelInfoResponse(
        model_name="NLLB-200 (No Language Left Behind)",
        model_type="Sequence-to-Sequence Multilingual Transformer",
        num_parameters="615 Million (Distilled)",
        supported_languages_count=len(get_supported_languages()),
        framework="PyTorch + Hugging Face Transformers",
        device=device_info.get("device_name", "CPU"),
        license="CC-BY-NC 4.0 (Non-Commercial Research & Educational Use)",
        attribution="NALL-200 Team, Meta AI Research (2022). No Language Left Behind: Scaling Human-Centered Machine Translation.",
        limitations=[
            "Low-resource language translations may occasionally contain hallucinated named entities.",
            "Idiomatic expressions and colloquial slang might be translated literally.",
            "Maximum recommended input length per segment is 1024 tokens to avoid degradation.",
            "Not certified for critical legal, medical, or life-safety translation."
        ]
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_translation_analytics(db: Session = Depends(get_db)):
    """Computes translation analytics from actual SQLite history."""
    total = db.query(func.count(TranslationHistory.id)).scalar() or 0
    avg_latency = db.query(func.avg(TranslationHistory.inference_time)).scalar() or 0.0

    # Top source languages
    src_counts = db.query(
        TranslationHistory.source_language_name,
        func.count(TranslationHistory.id).label("count")
    ).group_by(TranslationHistory.source_language_name).order_by(desc("count")).limit(5).all()

    top_sources = [{"language": row[0] or "Unknown", "count": row[1]} for row in src_counts]

    # Top target languages
    tgt_counts = db.query(
        TranslationHistory.target_language_name,
        func.count(TranslationHistory.id).label("count")
    ).group_by(TranslationHistory.target_language_name).order_by(desc("count")).limit(5).all()

    top_targets = [{"language": row[0] or "Unknown", "count": row[1]} for row in tgt_counts]

    # Recent activity
    recent_items = db.query(TranslationHistory).order_by(desc(TranslationHistory.created_at)).limit(5).all()
    recent_activity = [
        HistoryItem(
            id=i.id,
            original_text=i.source_text,
            translated_text=i.translated_text,
            source_language=i.source_language,
            source_language_name=i.source_language_name,
            target_language=i.target_language,
            target_language_name=i.target_language_name,
            detected_language=i.detected_language,
            inference_time=i.inference_time,
            model_name=i.model_name,
            created_at=i.created_at
        ) for i in recent_items
    ]

    return AnalyticsResponse(
        total_translations=total,
        average_inference_time_ms=round(avg_latency * 1000, 1),
        top_source_languages=top_sources,
        top_target_languages=top_targets,
        recent_activity=recent_activity
    )


@router.get("/lab/info")
def get_deep_learning_lab_info():
    """Provides architectural metrics, training history, and evaluation data for the DL Lab."""
    # Read custom transformer history if available
    custom_history = None
    for history_file in [
        "ml_lab/evaluation/results/custom_transformer_history.json",
        "evaluation/results/custom_transformer_history.json"
    ]:
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    custom_history = json.load(f)
                break
            except Exception:
                pass

    # Read evaluation benchmark results if available
    benchmark_results = None
    for eval_file in [
        "ml_lab/evaluation/results/evaluation_results.json",
        "evaluation/results/evaluation_results.json"
    ]:
        if os.path.exists(eval_file):
            try:
                with open(eval_file, "r") as f:
                    benchmark_results = json.load(f)
                break
            except Exception:
                pass

    return {
        "custom_transformer": {
            "name": "Attention Is All You Need (PyTorch Implementation)",
            "encoder_layers": 2,
            "decoder_layers": 2,
            "attention_heads": 4,
            "d_model": 128,
            "d_ff": 256,
            "training_history": custom_history
        },
        "evaluation_benchmark": benchmark_results,
        "transformer_components": [
            {
                "name": "Multi-Head Attention",
                "formula": "MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O",
                "description": "Enables the model to jointly attend to information from different representation subspaces at different positions."
            },
            {
                "name": "Scaled Dot-Product Attention",
                "formula": "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V",
                "description": "Calculates normalized similarity weights between query and key tokens, scaling by sqrt(d_k) to prevent vanishing gradients."
            },
            {
                "name": "Sinusoidal Positional Encoding",
                "formula": "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))",
                "description": "Injects sequence position awareness without requiring recurrence or convolutional layers."
            },
            {
                "name": "Residual Connections & LayerNorm",
                "formula": "Output = LayerNorm(x + Sublayer(x))",
                "description": "Stabilizes deep network optimization, enabling smooth gradient flow across multiple attention layers."
            }
        ]
    }
