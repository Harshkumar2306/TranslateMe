"""
FastAPI REST API Routes for TranslateMe
Exclusively powered by Custom Multilingual PyTorch Transformer built from scratch.
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.database.session import get_db
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
from backend.app.services.custom_multilingual_service import custom_multilingual_engine, CODE_MAP, REVERSE_MAP
from backend.app.services.language_detector import detect_language
from backend.app.services.document_service import extract_text_from_file, translate_document
from backend.app.core.device import get_device_info
from ml_lab.tokenizer import list_supported_languages, is_supported_language

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint displaying system and device status."""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "service": "TranslateMe Custom Multilingual Translation API",
        "model_engine": "Custom PyTorch Transformer (From Scratch)",
        "model_loaded": custom_multilingual_engine.is_loaded,
        "device": device_info
    }


@router.get("/languages", response_model=List[LanguageInfo])
def list_languages():
    """Returns the list of all languages supported by the custom Transformer."""
    return [
        LanguageInfo(
            code="en",
            name="English",
            native_name="English",
            script="Latin",
            iso="en-US"
        ),
        LanguageInfo(
            code="hi",
            name="Hindi",
            native_name="हिन्दी",
            script="Devanagari",
            iso="hi-IN"
        ),
        LanguageInfo(
            code="ta",
            name="Tamil",
            native_name="தமிழ்",
            script="Tamil",
            iso="ta-IN"
        ),
        LanguageInfo(
            code="te",
            name="Telugu",
            native_name="తెలుగు",
            script="Telugu",
            iso="te-IN"
        )
    ]


@router.post("/detect-language", response_model=DetectLanguageResponse)
def detect_lang(req: DetectLanguageRequest):
    """Detects the language of the provided text."""
    code, name, conf = detect_language(req.text)
    norm_code = CODE_MAP.get(code, "en")
    return DetectLanguageResponse(
        detected_code=norm_code,
        detected_name=REVERSE_MAP.get(norm_code, name),
        confidence=conf
    )


@router.post("/translate", response_model=TranslateResponse)
def translate_text(req: TranslateRequest, db: Session = Depends(get_db)):
    """Translates text between any supported language pair using our custom Transformer."""
    try:
        result = custom_multilingual_engine.translate(
            text=req.text,
            source_lang=req.source_language,
            target_lang=req.target_language,
            num_beams=req.num_beams or 1,
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


@router.post("/lab/attention")
def extract_live_attention(payload: Dict[str, Any]):
    """
    Extracts real live Cross-Attention weights from the custom Transformer
    for interactive heatmaps in the Deep Learning Lab.
    """
    text = payload.get("text", "Hello, how are you today?")
    source_lang = payload.get("source_lang", "en")
    target_lang = payload.get("target_lang", "hi")

    try:
        result = custom_multilingual_engine.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            num_beams=1,
            return_attention=True
        )
        return {
            "source_text": text,
            "translated_text": result["translated_text"],
            "source_tokens": result.get("source_subwords", text.split()),
            "target_tokens": result.get("target_subwords", result["translated_text"].split()),
            "attention_matrix": result.get("cross_attention_matrix", [])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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


@router.post("/document/translate", response_model=DocumentTranslateResponse)
async def translate_doc_endpoint(
    file: UploadFile = File(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    """
    Extracts text from uploaded PDF or TXT document and translates paragraph by paragraph
    using our custom Transformer, preserving structure and enforcing upload size limits.
    """
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file size ({len(content) / (1024*1024):.1f}MB) exceeds the 10MB limit."
        )

    filename = file.filename or "uploaded_file"
    paragraphs = extract_text_from_file(file, content)
    full_translated, elapsed = translate_document(
        paragraphs=paragraphs,
        source_lang=source_language,
        target_lang=target_language,
        max_paragraphs=50
    )

    return DocumentTranslateResponse(
        filename=filename,
        total_paragraphs=len(paragraphs),
        translated_text=full_translated,
        inference_time=elapsed
    )


@router.get("/analytics", response_model=AnalyticsResponse)
def get_analytics(db: Session = Depends(get_db)):
    """Aggregates metrics for the Analytics dashboard."""
    total = db.query(TranslationHistory).count()
    avg_latency = db.query(func.avg(TranslationHistory.inference_time)).scalar() or 0.0

    src_counts = db.query(
        TranslationHistory.source_language_name,
        func.count(TranslationHistory.id).label("count")
    ).group_by(TranslationHistory.source_language_name).order_by(desc("count")).limit(5).all()

    top_sources = [{"language": row[0] or "Unknown", "count": row[1]} for row in src_counts]

    tgt_counts = db.query(
        TranslationHistory.target_language_name,
        func.count(TranslationHistory.id).label("count")
    ).group_by(TranslationHistory.target_language_name).order_by(desc("count")).limit(5).all()

    top_targets = [{"language": row[0] or "Unknown", "count": row[1]} for row in tgt_counts]

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
    """Provides architectural metrics, real training history, and scientific evaluation data."""
    history_file = "ml_lab/evaluation/results/custom_multilingual_training_history.json"
    custom_history = None
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                custom_history = json.load(f)
        except Exception:
            pass

    eval_file = "ml_lab/evaluation/results/custom_multilingual_evaluation_report.json"
    eval_report = None
    if os.path.exists(eval_file):
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_report = json.load(f)
        except Exception:
            pass

    return {
        "custom_transformer": {
            "name": "TranslateMe Custom Multilingual Transformer",
            "encoder_layers": 3,
            "decoder_layers": 3,
            "attention_heads": 4,
            "d_model": 256,
            "d_ff": 1024,
            "parameters": 6033408,
            "shared_vocab_size": 2000,
            "training_history": custom_history
        },
        "evaluation_benchmark": eval_report,
        "transformer_components": [
            {
                "name": "Multi-Head Attention",
                "formula": "MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O",
                "description": "Jointly attends to information from multiple representation subspaces."
            },
            {
                "name": "Scaled Dot-Product Attention",
                "formula": "Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V",
                "description": "Calculates normalized similarity weights between query and key tokens, scaled by sqrt(d_k)."
            },
            {
                "name": "Sinusoidal Positional Encoding",
                "formula": "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))",
                "description": "Injects sequence position awareness into token vectors."
            },
            {
                "name": "Residual Connections & LayerNorm",
                "formula": "Output = LayerNorm(x + Sublayer(x))",
                "description": "Enables smooth gradient flow across deep encoder and decoder stacks."
            }
        ]
    }


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_specifications():
    """Returns technical specifications of our custom-trained Multilingual Transformer."""
    return ModelInfoResponse(
        model_name="TranslateMe Custom Multilingual Transformer",
        model_type="Seq2Seq Transformer (Encoder-Decoder)",
        num_parameters="27,974,144 (27.97M)",
        parameters="27,974,144 (27.97M)",
        layers=6,
        attention_heads=6,
        d_model=384,
        d_ff=1536,
        vocab_size="8,192 subwords",
        tokenizer_type="Byte-Pair Encoding (BPE) Multilingual Subword Tokenizer (ByteLevel)",
        supported_languages_count=4,
        framework="Pure PyTorch (Vaswani et al., 2017)",
        device=str(custom_multilingual_engine.device),
        license="MIT Open Source (Own Independent Implementation)",
        attribution="Trained completely from scratch with PyTorch. No pretrained translation weights.",
        limitations=[
            "Trained on curated 4-way multilingual parallel corpus (OPUS, TED2020, QED)",
            "Vocabulary size is 8,192 subwords via custom ByteLevel BPE tokenizer",
            "Context length is optimized up to 128 subwords",
            "Full seq2seq Transformer trained on Dual Tesla T4 GPUs with PyTorch AMP FP16"
        ]
    )
